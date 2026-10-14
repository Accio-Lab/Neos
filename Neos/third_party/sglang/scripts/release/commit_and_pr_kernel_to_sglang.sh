#!/bin/bash
set -e

                                                                                  
                                                                         
 
            
                                                   
                                               

KERNEL_VERSION="$1"
BRANCH_NAME="$2"

if [ -z "$KERNEL_VERSION" ] || [ -z "$BRANCH_NAME" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <kernel_version> <branch_name>"
    exit 1
fi

                                   
echo "Getting changed files..."
FILES_LIST=$(git diff --name-only | sed 's/^/- /')
COMMIT_FILES=$(git diff --name-only | sed 's/^/          - /')

                
echo "Committing changes..."
git add -A
git commit -m "chore: bump sgl-kernel version to ${KERNEL_VERSION} in SGLang

This commit updates the sgl-kernel version across SGLang files to match
the version defined in sgl-kernel/pyproject.toml.

Files updated:
${COMMIT_FILES}

🤖 Generated with GitHub Actions"

              
echo "Pushing to ${BRANCH_NAME}..."
git push origin "${BRANCH_NAME}"

                     
echo "Creating pull request..."
PR_URL=$(gh pr create \
  --title "chore: bump sgl-kernel version to ${KERNEL_VERSION}" \
  --body "## Summary

This PR bumps the \`sgl-kernel\` version to \`${KERNEL_VERSION}\` across SGLang files to match the version defined in \`sgl-kernel/pyproject.toml\`.

**Kernel Version:** \`${KERNEL_VERSION}\`

## Files Updated
${FILES_LIST}

## Context

The sgl-kernel version in \`sgl-kernel/pyproject.toml\` has been updated. This PR ensures that all SGLang files referencing the kernel version are updated accordingly:
- \`python/pyproject.toml\` - dependency specification
- \`python/sglang/srt/entrypoints/engine.py\` - version check
- \`docker/Dockerfile\` - Docker build argument

🤖 Generated with GitHub Actions" \
  --base main \
  --head "${BRANCH_NAME}")

echo "✓ Pull request created successfully"

                                
if [ -n "$GITHUB_STEP_SUMMARY" ]; then
  cat >> "$GITHUB_STEP_SUMMARY" <<EOF
## ✅ Kernel Version Bump Complete

**Kernel Version:** \`${KERNEL_VERSION}\`

### 📝 Pull Request Created
${PR_URL}

### 📦 Files Updated
${FILES_LIST}
EOF
fi
