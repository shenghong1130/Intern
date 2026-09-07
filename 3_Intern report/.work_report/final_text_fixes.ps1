$ErrorActionPreference = 'Stop'
$path = 'D:\我\大学\Intern\3_Intern report\实习报告_机器人与KV260共享计算平台_许胜宏_最终版.docx'
$word = New-Object -ComObject Word.Application
try {
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($path, $false, $false)
    function Replace-Text([string]$findText, [string]$replacement) {
        $r = $doc.Content.Duplicate
        $f = $r.Find
        $f.ClearFormatting()
        $f.Replacement.ClearFormatting()
        [void]$f.Execute($findText,$false,$false,$false,$false,$false,$true,1,$false,$replacement,2)
    }
    Replace-Text 'TonyPi；AprilTag；路径规划；KV260；PYNQ；FPGA；FastAPI；资源调度；嵌入式系统；机器人竞赛' 'TonyPi；AprilTag；机器人定位与导航；Motion-Aware A*；NFC通信；KV260；PYNQ；FPGA共享计算平台；Central Server；多节点调度'
    Replace-Text '表 4-1 Worker 主要状态' '表 4-3 Worker 主要状态'
    Replace-Text '表 4-2 Central Admin Dashboard 功能' '表 4-4 Central Admin Dashboard 功能'
    $doc.Save()
    Write-Output ('PAGES=' + $doc.ComputeStatistics(2))
    Write-Output ('WORDS=' + $doc.ComputeStatistics(0))
    $doc.Close($true)
}
finally {
    if ($word) { $word.Quit() }
}

