$cache = Join-Path (Get-Location) '.vector_cache_new2\glove.6B.zip'
$model = Join-Path (Get-Location) 'poem_hans_app\models\best_han_model.pt'
$prevModelSize = -1
while ($true) {
    if (Test-Path $cache) {
        $g = Get-Item $cache
        Write-Output "[monitor] GloVe: $($g.Length) bytes, modified: $($g.LastWriteTime)"
    } else {
        Write-Output "[monitor] GloVe: not present"
    }
    if (Test-Path $model) {
        $m = Get-Item $model
        if ($m.Length -ne $prevModelSize) {
            Write-Output "[monitor] Model updated: $($m.Length) bytes, modified: $($m.LastWriteTime)"
            $prevModelSize = $m.Length
        }
    } else {
        Write-Output "[monitor] Model: not present"
    }
    $procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like '*poem_hans_app.src.train*' }
    if (-not $procs) {
        Write-Output "[monitor] No active training process detected."
        if (Test-Path $model) {
            Write-Output "[monitor] Training appears finished."
            break
        }
    }
    Start-Sleep -Seconds 30
}
Write-Output "[monitor] Monitor exiting."