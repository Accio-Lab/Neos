//! mTLS (Mutual TLS) integration tests
//!
//! Tests for TLS and mTLS communication between router and workers.
//! Covers:
//! - Successful mTLS communication with client certificates
//! - TLS failure without client certificate when required
//! - TLS-only mode (server authentication only)
//! - TLS failure without CA certificate

use std::{io::BufReader, time::Duration};

use rustls::{ClientConfig, RootCertStore};
use rustls_pemfile::{certs, pkcs8_private_keys};
use serde_json::json;

use crate::common::{
    test_certs::TestCertificates,
    tls_mock_worker::{TlsMockWorker, TlsMockWorkerConfig},
};

                                      
fn ensure_crypto_provider() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Helper to create a TLS-enabled reqwest client with client certificates
fn create_mtls_client(
    test_certs: &TestCertificates,
) -> Result<reqwest::Client, Box<dyn std::error::Error>> {
    ensure_crypto_provider();

                          
    let ca_file = std::fs::File::open(&test_certs.ca_cert_path)?;
    let mut ca_reader = BufReader::new(ca_file);
    let ca_certs: Vec<_> = certs(&mut ca_reader).filter_map(|r| r.ok()).collect();

    let mut root_store = RootCertStore::empty();
    for cert in ca_certs {
        root_store.add(cert)?;
    }

                              
    let cert_file = std::fs::File::open(&test_certs.client_cert_path)?;
    let mut cert_reader = BufReader::new(cert_file);
    let client_certs: Vec<_> = certs(&mut cert_reader).filter_map(|r| r.ok()).collect();

                      
    let key_file = std::fs::File::open(&test_certs.client_key_path)?;
    let mut key_reader = BufReader::new(key_file);
    let client_key = pkcs8_private_keys(&mut key_reader)
        .next()
        .ok_or("No private key found")??;

                                                  
    let client_config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_client_auth_cert(client_certs, client_key.into())
        .map_err(|e| format!("Failed to build client config: {}", e))?;

    let client = reqwest::Client::builder()
        .use_preconfigured_tls(client_config)
        .timeout(Duration::from_secs(10))
        .build()?;

    Ok(client)
}

/// Helper to create a TLS client without client certificates (for server-auth-only)
fn create_tls_client(
    test_certs: &TestCertificates,
) -> Result<reqwest::Client, Box<dyn std::error::Error>> {
    ensure_crypto_provider();

                          
    let ca_file = std::fs::File::open(&test_certs.ca_cert_path)?;
    let mut ca_reader = BufReader::new(ca_file);
    let ca_certs: Vec<_> = certs(&mut ca_reader).filter_map(|r| r.ok()).collect();

    let mut root_store = RootCertStore::empty();
    for cert in ca_certs {
        root_store.add(cert)?;
    }

                                                     
    let client_config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    let client = reqwest::Client::builder()
        .use_preconfigured_tls(client_config)
        .timeout(Duration::from_secs(10))
        .build()?;

    Ok(client)
}

/// Helper to create a client without CA certificate (for failure test)
fn create_client_without_ca() -> Result<reqwest::Client, Box<dyn std::error::Error>> {
    ensure_crypto_provider();

                                                                      
    let root_store = RootCertStore::empty();
    let client_config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    let client = reqwest::Client::builder()
        .use_preconfigured_tls(client_config)
        .timeout(Duration::from_secs(10))
        .build()?;

    Ok(client)
}

#[cfg(test)]
mod mtls_tests {
    use super::*;

    /// Test successful mTLS communication between client and TLS worker
    ///
    /// This test verifies that:
    /// 1. TLS mock worker starts with mTLS configuration
    /// 2. Client with proper certificates can connect and communicate
    /// 3. Requests succeed with proper authentication
    #[tokio::test]
    async fn test_mtls_successful_communication() {
                                     
        let certs = TestCertificates::generate().expect("Failed to generate test certificates");

                                         
        let mut worker = TlsMockWorker::new(TlsMockWorkerConfig {
            port: 0,                    
            require_client_cert: true,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });

        let worker_url = worker
            .start(
                &certs.server_cert_path,
                &certs.server_key_path,
                Some(&certs.ca_cert_path),
            )
            .await
            .expect("Failed to start mTLS worker");

                                                   
        let client = create_mtls_client(&certs).expect("Failed to create mTLS client");

                               
        let health_resp = client
            .get(format!("{}/health", worker_url))
            .send()
            .await
            .expect("Health request failed");

        assert!(
            health_resp.status().is_success(),
            "Health check should succeed with mTLS: {}",
            health_resp.status()
        );

        let health_json: serde_json::Value = health_resp.json().await.unwrap();
        assert_eq!(health_json["status"], "healthy");
        assert_eq!(health_json["tls_enabled"], true);

                                 
        let payload = json!({
            "text": "Test mTLS request",
            "stream": false
        });

        let gen_resp = client
            .post(format!("{}/generate", worker_url))
            .json(&payload)
            .send()
            .await
            .expect("Generate request failed");

        assert!(
            gen_resp.status().is_success(),
            "Generate should succeed with mTLS: {}",
            gen_resp.status()
        );

        let gen_json: serde_json::Value = gen_resp.json().await.unwrap();
        assert!(gen_json["text"].as_str().is_some());
        assert_eq!(gen_json["meta_info"]["tls_verified"], true);

                  
        worker.stop().await;
    }

    /// Test that mTLS worker rejects connections without client certificate
    ///
    /// This test verifies that:
    /// 1. mTLS worker requires client certificate
    /// 2. Connection without client cert fails
    #[tokio::test]
    async fn test_mtls_failure_without_client_cert() {
                                     
        let certs = TestCertificates::generate().expect("Failed to generate test certificates");

                                                                
        let mut worker = TlsMockWorker::new(TlsMockWorkerConfig {
            port: 0,
            require_client_cert: true,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });

        let worker_url = worker
            .start(
                &certs.server_cert_path,
                &certs.server_key_path,
                Some(&certs.ca_cert_path),
            )
            .await
            .expect("Failed to start mTLS worker");

                                                                                         
        let client = create_tls_client(&certs).expect("Failed to create TLS-only client");

                                                                           
        let result = client.get(format!("{}/health", worker_url)).send().await;

                                                
        assert!(
            result.is_err(),
            "Connection should fail without client certificate"
        );

                  
        worker.stop().await;
    }

    /// Test TLS-only mode (server authentication only, no client cert required)
    ///
    /// This test verifies that:
    /// 1. TLS worker can operate without requiring client certificates
    /// 2. Client can connect with just CA certificate for server verification
    #[tokio::test]
    async fn test_tls_server_auth_only() {
                                     
        let certs = TestCertificates::generate().expect("Failed to generate test certificates");

                                                                    
        let mut worker = TlsMockWorker::new(TlsMockWorkerConfig {
            port: 0,
            require_client_cert: false,                 
            response_delay_ms: 0,
            fail_rate: 0.0,
        });

        let worker_url = worker
            .start(
                &certs.server_cert_path,
                &certs.server_key_path,
                None,                                        
            )
            .await
            .expect("Failed to start TLS worker");

                                                                                   
        let client = create_tls_client(&certs).expect("Failed to create TLS client");

                                                                    
        let health_resp = client
            .get(format!("{}/health", worker_url))
            .send()
            .await
            .expect("Health request failed");

        assert!(
            health_resp.status().is_success(),
            "Health check should succeed in TLS-only mode: {}",
            health_resp.status()
        );

        let health_json: serde_json::Value = health_resp.json().await.unwrap();
        assert_eq!(health_json["status"], "healthy");

                                         
        let payload = json!({
            "model": "mock-tls-model",
            "messages": [{"role": "user", "content": "Hello TLS"}]
        });

        let chat_resp = client
            .post(format!("{}/v1/chat/completions", worker_url))
            .json(&payload)
            .send()
            .await
            .expect("Chat request failed");

        assert!(
            chat_resp.status().is_success(),
            "Chat should succeed in TLS-only mode: {}",
            chat_resp.status()
        );

                  
        worker.stop().await;
    }

    /// Test TLS failure when client doesn't have CA certificate
    ///
    /// This test verifies that:
    /// 1. Client cannot verify server without proper CA certificate
    /// 2. Connection fails due to certificate verification
    #[tokio::test]
    async fn test_tls_failure_without_ca_cert() {
                                     
        let certs = TestCertificates::generate().expect("Failed to generate test certificates");

                                
        let mut worker = TlsMockWorker::new(TlsMockWorkerConfig {
            port: 0,
            require_client_cert: false,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });

        let worker_url = worker
            .start(&certs.server_cert_path, &certs.server_key_path, None)
            .await
            .expect("Failed to start TLS worker");

                                               
        let client = create_client_without_ca().expect("Failed to create client without CA");

                                                                             
        let result = client.get(format!("{}/health", worker_url)).send().await;

                                                                 
        assert!(
            result.is_err(),
            "Connection should fail without CA certificate for server verification"
        );

                  
        worker.stop().await;
    }

    /// Test multiple concurrent mTLS requests
    ///
    /// This test verifies that mTLS connections work correctly under concurrent load
    #[tokio::test]
    async fn test_mtls_concurrent_requests() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

                                     
        let certs = TestCertificates::generate().expect("Failed to generate test certificates");

                                         
        let mut worker = TlsMockWorker::new(TlsMockWorkerConfig {
            port: 0,
            require_client_cert: true,
            response_delay_ms: 10,                                
            fail_rate: 0.0,
        });

        let worker_url = worker
            .start(
                &certs.server_cert_path,
                &certs.server_key_path,
                Some(&certs.ca_cert_path),
            )
            .await
            .expect("Failed to start mTLS worker");

                             
        let client = Arc::new(create_mtls_client(&certs).expect("Failed to create mTLS client"));
        let success_count = Arc::new(AtomicUsize::new(0));
        let worker_url = Arc::new(worker_url);

        let mut handles = Vec::new();

                                    
        for i in 0..10 {
            let client_clone = Arc::clone(&client);
            let success_clone = Arc::clone(&success_count);
            let url_clone = Arc::clone(&worker_url);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Concurrent mTLS request {}", i),
                    "stream": false
                });

                let resp = client_clone
                    .post(format!("{}/generate", url_clone))
                    .json(&payload)
                    .send()
                    .await;

                if let Ok(response) = resp {
                    if response.status().is_success() {
                        success_clone.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });

            handles.push(handle);
        }

                                            
        for handle in handles {
            handle.await.unwrap();
        }

                                      
        assert_eq!(
            success_count.load(Ordering::SeqCst),
            10,
            "All concurrent mTLS requests should succeed"
        );

                  
        worker.stop().await;
    }
}

#[cfg(test)]
mod certificate_tests {
    use super::*;

    /// Test that certificate generation works correctly
    #[test]
    fn test_certificate_generation() {
        let certs = TestCertificates::generate().expect("Failed to generate certificates");

                                                   
        assert!(certs.ca_cert_path.exists());
        assert!(certs.ca_key_path.exists());
        assert!(certs.server_cert_path.exists());
        assert!(certs.server_key_path.exists());
        assert!(certs.client_cert_path.exists());
        assert!(certs.client_key_path.exists());

                            
        let ca_cert_content = std::fs::read_to_string(&certs.ca_cert_path).unwrap();
        assert!(ca_cert_content.contains("-----BEGIN CERTIFICATE-----"));
        assert!(ca_cert_content.contains("-----END CERTIFICATE-----"));

        let server_key_content = std::fs::read_to_string(&certs.server_key_path).unwrap();
        assert!(server_key_content.contains("PRIVATE_KEY_FROM_ENV"));
    }

    /// Test that generated certificates can be parsed by rustls
    #[test]
    fn test_certificate_parsing() {
        ensure_crypto_provider();

        let test_certs = TestCertificates::generate().expect("Failed to generate certificates");

                                                        
        let ca_file = std::fs::File::open(&test_certs.ca_cert_path).unwrap();
        let mut ca_reader = BufReader::new(ca_file);
        let ca_certs: Vec<_> = certs(&mut ca_reader).filter_map(|r| r.ok()).collect();
        assert!(!ca_certs.is_empty(), "CA certificate should be parseable");

                                                  
        let cert_file = std::fs::File::open(&test_certs.client_cert_path).unwrap();
        let mut cert_reader = BufReader::new(cert_file);
        let client_certs: Vec<_> = certs(&mut cert_reader).filter_map(|r| r.ok()).collect();
        assert!(
            !client_certs.is_empty(),
            "Client certificate should be parseable"
        );

                                          
        let key_file = std::fs::File::open(&test_certs.client_key_path).unwrap();
        let mut key_reader = BufReader::new(key_file);
        let keys: Vec<_> = pkcs8_private_keys(&mut key_reader)
            .filter_map(|r| r.ok())
            .collect();
        assert!(!keys.is_empty(), "Client key should be parseable");

                                                       
        let mut root_store = RootCertStore::empty();
        for cert in ca_certs {
            root_store
                .add(cert)
                .expect("Should be able to add CA cert to store");
        }

        let key = keys.into_iter().next().unwrap();
        let config_result = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_client_auth_cert(client_certs, key.into());

        assert!(
            config_result.is_ok(),
            "Should be able to create client config with certs"
        );
    }
}
