$ErrorActionPreference = 'Stop'
$path = 'D:\我\大学\Intern\3_Intern report\实习报告_机器人与KV260共享计算平台_许胜宏_最终版.docx'
$wdStyleNormal = -1
$wdStyleTOCHeading = -267
$word = New-Object -ComObject Word.Application
try {
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($path, $false, $false)
    foreach ($table in $doc.Tables) {
        foreach ($row in $table.Rows) {
            foreach ($cell in $row.Cells) {
                foreach ($p in $cell.Range.Paragraphs) {
                    $p.Style = $wdStyleNormal
                    $p.Format.FirstLineIndent = 0
                    $p.Format.SpaceAfter = 0
                    $p.Format.LineSpacingRule = 0
                }
            }
        }
    }
    foreach ($p in $doc.Paragraphs) {
        $text = $p.Range.Text.Trim([char]13,[char]12,[char]7,[char]32,[char]9)
        if (-not $text -and $p.OutlineLevel -ge 1 -and $p.OutlineLevel -le 3) {
            $p.Style = $wdStyleNormal
            $p.Format.PageBreakBefore = 0
            $p.Format.OutlineLevel = 9
        }
    }
    foreach ($label in @('摘要','关键词','目录')) {
        foreach ($p in $doc.Paragraphs) {
            if ($p.Range.Text.Trim([char]13,[char]12,[char]7,[char]32,[char]9) -eq $label) {
                if ($label -eq '目录') { $p.Style = $wdStyleTOCHeading }
                else { try { $p.Style = 'FrontHeading' } catch { $p.Style = $wdStyleNormal } }
                $p.Format.OutlineLevel = 9
                if ($label -eq '摘要') { $p.Format.PageBreakBefore = -1 }
                break
            }
        }
    }
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    foreach ($p in $doc.Paragraphs) {
        if ($p.Range.Text.Trim([char]13,[char]12,[char]7,[char]32,[char]9) -eq '目录') {
            $p.Style = $wdStyleTOCHeading
            $p.Format.OutlineLevel = 9
            break
        }
    }
    $doc.Save()
    Write-Output ('PAGES=' + $doc.ComputeStatistics(2))
    Write-Output ('WORDS=' + $doc.ComputeStatistics(0))
    Write-Output ('TABLES=' + $doc.Tables.Count)
    Write-Output ('TOCS=' + $doc.TablesOfContents.Count)
    $doc.Close($true)
}
finally {
    if ($word) { $word.Quit() }
}



