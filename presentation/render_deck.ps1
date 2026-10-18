# render_deck.ps1 -- export project-overview.pptx to PDF and one PNG per slide using PowerPoint (Windows only).
#
#   powershell -File presentation\render_deck.ps1 [-OutDir <folder for the PNGs>]
#
# The PDF is written next to the .pptx; the PNGs go to -OutDir (default: %TEMP%\deck_render), so no build output is
# added to the repository.
param(
    [string]$Pptx = (Join-Path $PSScriptRoot 'project-overview.pptx'),
    [string]$OutDir = (Join-Path $env:TEMP 'deck_render')
)

$pptxFull = (Resolve-Path $Pptx).Path
$pdf = [System.IO.Path]::ChangeExtension($pptxFull, '.pdf')
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Get-ChildItem $OutDir -Filter 'Slide*.PNG' -ErrorAction SilentlyContinue | Remove-Item -Force

$app = New-Object -ComObject PowerPoint.Application
try {
    # Open(FileName, ReadOnly, Untitled, WithWindow)
    $pres = $app.Presentations.Open($pptxFull, -1, 0, 0)
    $pres.SaveAs($pdf, 32)                       # ppSaveAsPDF
    $pres.Export($OutDir, 'PNG', 1920, 1080)
    $pres.Close()
}
finally {
    $app.Quit()
}
Write-Output "PDF : $pdf"
Write-Output "PNGs: $OutDir"
