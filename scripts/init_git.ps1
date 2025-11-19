# Initialize git repository and optionally add remote
param(
    [string]$remoteUrl
)

git init
git add -A
git commit -m "Initial commit: starter project"
if ($remoteUrl) {
    git remote add origin $remoteUrl
    Write-Output "Added remote $remoteUrl"
}

Write-Output "Repository initialized. Edit README.md and push when ready."
