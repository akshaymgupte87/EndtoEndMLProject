param(
  [int]$UserIndex = 0,
  [int]$Limit = 99,
  [string]$ApiUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Stop"
Write-Host "Stop only the local Redis container: docker compose stop redis"
Write-Host "Request a limit that has not warmed the in-memory cache. API should still respond."
$response = Invoke-RestMethod -Uri "$ApiUrl/recommendations/$UserIndex`?limit=$Limit"
if (-not $response.items) {
  Write-Error "Expected recommendations even though Redis is unavailable."
}
Write-Host "API stayed available with $($response.items.Count) recommendations."
Write-Host "Check $ApiUrl/metrics for recommendation_cache_errors_total, then recover: docker compose start redis"
