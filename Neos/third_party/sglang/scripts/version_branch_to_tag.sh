#!/bin/bash
set -euxo pipefail

                                  
                                                                                 
                                                                                                   

git fetch origin --prune

                                     
branches=$(git branch -r | grep 'origin/v' | sed 's/origin\///')

                          
for branch in $branches; do
    echo "Processing branch: $branch"

                                        
    commit_hash=$(git rev-parse origin/$branch)

                                                                         
    git tag $branch $commit_hash

                                       
    git push origin --delete $branch
done

                                        
git push --tags

echo "All branches starting with 'v' have been tagged, deleted from remote, and pushed to the remote repository."
