#!/bin/bash
set -e

                                                                 
                                                                    
 
            
                                          
                                       
                                               

VERSION_TYPE="$1"
NEW_VERSION="$2"
BRANCH_NAME="$3"

if [ -z "$VERSION_TYPE" ] || [ -z "$NEW_VERSION" ] || [ -z "$BRANCH_NAME" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <version_type> <new_version> <branch_name>"
    exit 1
fi

                                   
echo "Getting changed files..."
FILES_LIST=$(git diff --name-only | sed 's/^/- /')
COMMIT_FILES=$(git diff --name-only | sed 's/^/          - /')

                
echo "Committing changes..."
git add -A
git commit -m "chore: bump ${VERSION_TYPE} version to ${NEW_VERSION}

This commit updates the ${VERSION_TYPE} version across all relevant files:
${COMMIT_FILES}

🤖 Generated with GitHub Actions"

              
echo "Pushing to ${BRANCH_NAME}..."
git push origin "${BRANCH_NAME}"

                     
echo "Creating pull request..."
PR_URL=$(gh pr create \
  --title "chore: bump ${VERSION_TYPE} version to ${NEW_VERSION}" \
  --body "## Summary

This PR bumps the ${VERSION_TYPE} version to \`${NEW_VERSION}\` across all relevant files.

## Files Updated
${FILES_LIST}

🤖 Generated with GitHub Actions" \
  --base main \
  --head "${BRANCH_NAME}")

echo "✓ Pull request created successfully"

                                
if [ -n "$GITHUB_STEP_SUMMARY" ]; then
  cat >> "$GITHUB_STEP_SUMMARY" <<EOF
## ✅ Version Bump Complete

**Version Type:** ${VERSION_TYPE}
**New Version:** \`${NEW_VERSION}\`

### 📝 Pull Request Created
${PR_URL}

### 📦 Files Updated
${FILES_LIST}
EOF
fi
