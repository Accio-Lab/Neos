//! Harmony model detection

use crate::core::{Worker, WorkerRegistry};

/// Harmony model detector
///
/// Detects if a model name indicates support for Harmony encoding/parsing.
pub(crate) struct HarmonyDetector;

impl HarmonyDetector {
    /// Check if a worker is a Harmony/GPT-OSS model.
    ///
    /// Detection priority:
    /// 1. Check if any model card has architectures containing "GptOssForCausalLM"
    /// 2. Check if any model card has hf_model_type equal to "gpt_oss"
    /// 3. Check if model_id contains "gpt-oss" substring (case-insensitive)
    pub fn is_harmony_worker(worker: &dyn Worker) -> bool {
        for model_card in worker.models() {
                                                           
            if model_card
                .architectures
                .iter()
                .any(|arch| arch == "GptOssForCausalLM")
            {
                return true;
            }

                                                 
            if let Some(ref model_type) = model_card.hf_model_type {
                if model_type == "gpt_oss" {
                    return true;
                }
            }

                                                      
            if Self::is_harmony_model(&model_card.id) {
                return true;
            }
        }

                                                     
        Self::is_harmony_model(worker.model_id())
    }

    /// Check if a model name contains "gpt-oss" (case-insensitive).
    pub fn is_harmony_model(model_name: &str) -> bool {
                                                                    
                                                                          
        model_name
            .as_bytes()
            .windows(7)                   
            .any(|window| window.eq_ignore_ascii_case(b"gpt-oss"))
    }

    /// Check if any worker for the given model is a Harmony/GPT-OSS worker.
    ///
    /// This method looks up workers from the registry by model name and checks
    /// if any of them are Harmony workers based on their metadata (architectures,
    /// hf_model_type).
    ///
    /// Falls back to string-based detection if no workers are registered for
    /// the model (e.g., during startup before workers are discovered).
    pub fn is_harmony_model_in_registry(registry: &WorkerRegistry, model_name: &str) -> bool {
                                     
        let workers = registry.get_by_model(model_name);

        if workers.is_empty() {
                                                                     
            return Self::is_harmony_model(model_name);
        }

                                                  
        workers
            .iter()
            .any(|worker| Self::is_harmony_worker(worker.as_ref()))
    }
}
