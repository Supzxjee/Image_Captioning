param([string]$DocumentPath)
$ErrorActionPreference = 'Stop'
$resolvedDocument = (Resolve-Path -LiteralPath $DocumentPath).Path
$renderDirectory = Join-Path (Split-Path -Parent $resolvedDocument) 'render'
New-Item -ItemType Directory -Path $renderDirectory -Force | Out-Null
$pdfPath = Join-Path $renderDirectory 'report.pdf'
$wordApplication = $null
$wordDocument = $null
try {
    $wordApplication = New-Object -ComObject Word.Application
    $wordApplication.Visible = $false
    $wordApplication.DisplayAlerts = 0
    $wordApplication.AutomationSecurity = 3
    $wordDocument = $wordApplication.Documents.Open($resolvedDocument, $false, $true)
    $wordDocument.Fields.Update() | Out-Null
    $wordDocument.Repaginate()
    $pageCount = $wordDocument.ComputeStatistics(2)
    $wordDocument.ExportAsFixedFormat($pdfPath, 17)
    Write-Output "Word export PASS: $pageCount pages; $pdfPath"
} finally {
    if ($null -ne $wordDocument) {
        $wordDocument.Close(0)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordDocument)
    }
    if ($null -ne $wordApplication) {
        $wordApplication.Quit()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApplication)
    }
}
