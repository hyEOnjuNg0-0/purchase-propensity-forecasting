Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $args -or $args.Count -eq 0) {
    Write-Error "사용법: .\\scripts\\run_ptf.ps1 <command> [args...]"
    exit 2
}

$command = [string]$args[0]
$arguments = @()
if ($args.Count -gt 1) {
    $arguments = $args[1..($args.Count - 1)]
}

# conda 환경을 매번 명시해 Codex/자동화 실행 시에도 동일 환경을 강제한다.
& conda run --no-capture-output -n ptf $command @arguments
$exitCode = $LASTEXITCODE

if ($null -eq $exitCode) {
    $exitCode = 0
}

exit $exitCode
