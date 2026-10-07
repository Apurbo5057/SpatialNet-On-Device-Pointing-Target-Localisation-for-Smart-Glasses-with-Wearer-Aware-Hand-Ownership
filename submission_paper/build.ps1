$ErrorActionPreference = 'Continue'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force qa | Out-Null
    & pdflatex -interaction=nonstopmode -halt-on-error main.tex *> qa/build-pass1.txt
    if ($LASTEXITCODE -ne 0) { throw 'First LaTeX pass failed; inspect qa/build-pass1.txt' }
    & bibtex main *> qa/bibtex.txt
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed; inspect qa/bibtex.txt' }
    & pdflatex -interaction=nonstopmode -halt-on-error main.tex *> qa/build-pass2.txt
    if ($LASTEXITCODE -ne 0) { throw 'Second LaTeX pass failed' }
    & pdflatex -interaction=nonstopmode -halt-on-error main.tex *> qa/build-pass3.txt
    if ($LASTEXITCODE -ne 0) { throw 'Final LaTeX pass failed' }
    Write-Output 'Built main.pdf. No model execution or experiment evaluation was performed.'
} finally {
    Pop-Location
}

