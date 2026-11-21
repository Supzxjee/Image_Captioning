$ErrorActionPreference = 'Stop'
$figureRoot = $PSScriptRoot
$chromePath = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
$figureSizes = @{
    '01_full_pipeline' = @(1840,1160)
    '02_M1_semantic_bank' = @(1150,810)
    '03_M2_bank_visual_alignment' = @(1760,1160)
    '04_M3_prompt_visual_alignment' = @(1640,820)
}
foreach ($figureName in ($figureSizes.Keys | Sort-Object)) {
    $figureSvg = Get-Content -LiteralPath (Join-Path $figureRoot "$figureName.svg") -Raw -Encoding UTF8
    $figureHtml = '<!doctype html><meta charset="utf-8"><style>html,body{margin:0;background:white}svg{display:block}</style>' + $figureSvg
    $htmlPath = Join-Path $figureRoot "$figureName.html"
    [IO.File]::WriteAllText($htmlPath, $figureHtml, [Text.UTF8Encoding]::new($false))
    $pngPath = Join-Path $figureRoot "$figureName.png"
    $figureUri = ([Uri]$htmlPath).AbsoluteUri
    $figureWidth,$figureHeight = $figureSizes[$figureName]
    $chromeArgs = @('--headless=new','--disable-gpu','--hide-scrollbars','--no-first-run', '--force-device-scale-factor=2',"--window-size=$figureWidth,$figureHeight",'--run-all-compositor-stages-before-draw',"--screenshot=`"$pngPath`"", "`"$figureUri`"")
    Start-Process -FilePath $chromePath -ArgumentList $chromeArgs -WindowStyle Hidden -Wait
    if (-not (Test-Path -LiteralPath $pngPath)) { throw "PNG not created: $figureName" }
    Write-Output "Rendered $figureName"
}
