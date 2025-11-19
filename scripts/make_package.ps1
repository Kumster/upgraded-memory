<#
Creates a folder `kitchen_ai_compass` under the current user's profile
and copies the repository contents into it. Excludes .git and .venv by default.

Usage:
  From project root: .\scripts\make_package.ps1
#>

$src = (Get-Location).Path
$dest = Join-Path $env:USERPROFILE 'kitchen_ai_compass'

Write-Output "Source: $src"
Write-Output "Destination: $dest"

if (Test-Path $dest) {
    Write-Output "Destination exists — removing: $dest"
    Remove-Item $dest -Recurse -Force
}

New-Item -ItemType Directory -Path $dest -Force | Out-Null

# Use robocopy for robust copying of files and dirs on Windows
$excludeDirs = @('.git', '.venv')
$excludeArg = '/XD ' + ($excludeDirs -join ' ')

Write-Output "Running robocopy... (this may print many lines)"
robocopy $src $dest *.* /E /COPY:DAT /R:1 /W:1 $excludeArg | Out-Null

Write-Output "Copy finished. Top-level of $dest:"
Get-ChildItem -Path $dest | Select-Object Name,Mode,Length
