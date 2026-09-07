$ErrorActionPreference = 'Stop'

$root = 'D:\我\大学\Intern'
$reportDir = Join-Path $root '3_Intern report'
$source = Join-Path $reportDir '实习报告_机器人与KV260共享计算平台_许胜宏.docx'
$backup = Join-Path $reportDir '实习报告_机器人与KV260共享计算平台_许胜宏_backup.docx'
$final = Join-Path $reportDir '实习报告_机器人与KV260共享计算平台_许胜宏_最终版.docx'
$work = Join-Path $reportDir '.work_report'
$images = Join-Path $reportDir 'images'

if (-not (Test-Path -LiteralPath $source)) { throw "Source not found: $source" }
if (-not (Test-Path -LiteralPath $backup)) { Copy-Item -LiteralPath $source -Destination $backup }
Copy-Item -LiteralPath $source -Destination $final -Force

$wdCollapseEnd = 0
$wdAlignParagraphCenter = 1
$wdAlignParagraphJustify = 3
$wdStyleNormal = -1
$wdStyleHeading1 = -2
$wdStyleHeading2 = -3
$wdStyleHeading3 = -4
$wdStyleCaption = -35
$wdPageBreak = 7
$wdAutoFitWindow = 2
$wdCellAlignVerticalCenter = 1
$wdPreferredWidthPoints = 3

function Find-Paragraph {
    param($doc, [string]$Text, [switch]$StartsWith)
    foreach ($p in $doc.Paragraphs) {
        $t = $p.Range.Text.Trim([char]13, [char]7, [char]32, [char]9)
        if (($StartsWith -and $t.StartsWith($Text)) -or ((-not $StartsWith) -and $t -eq $Text)) { return $p }
    }
    throw "Paragraph not found: $Text"
}

function Set-ParagraphText {
    param($Paragraph, [string]$Text)
    $r = $Paragraph.Range.Duplicate
    $r.End = $r.End - 1
    $r.Text = $Text
}

function Replace-AllText {
    param($doc, [string]$Find, [string]$Replace)
    $r = $doc.Content.Duplicate
    $f = $r.Find
    $f.ClearFormatting()
    $f.Replacement.ClearFormatting()
    [void]$f.Execute($Find, $false, $false, $false, $false, $false, $true, 1, $false, $Replace, 2)
}

function Apply-BodyFormat {
    param($Paragraph)
    $Paragraph.Alignment = $wdAlignParagraphJustify
    $Paragraph.Format.LineSpacingRule = 1
    $Paragraph.Format.SpaceAfter = 6
    $Paragraph.Format.SpaceBefore = 0
    $Paragraph.Format.FirstLineIndent = 24
    $Paragraph.Format.WidowControl = -1
}

function Insert-BlocksBeforeHeading {
    param($doc, [string]$TargetHeading, [array]$Blocks)
    $target = Find-Paragraph $doc $TargetHeading
    $pos = $target.Range.Start
    $text = (($Blocks | ForEach-Object { [string]$_.Text }) -join "`r") + "`r"
    $r = $doc.Range($pos, $pos)
    $r.InsertBefore($text)
    $inserted = $doc.Range($pos, $pos + $text.Length)
    $idx = 1
    foreach ($b in $Blocks) {
        $p = $inserted.Paragraphs.Item($idx)
        switch ([string]$b.Style) {
            'H1' { $p.Style = $wdStyleHeading1 }
            'H2' { $p.Style = $wdStyleHeading2 }
            'H3' { $p.Style = $wdStyleHeading3 }
            'Caption' { $p.Style = $wdStyleCaption; $p.Alignment = $wdAlignParagraphCenter; $p.Format.KeepWithNext = -1 }
            'NoIndent' { try { $p.Style = 'BodyNoIndent' } catch { $p.Style = $wdStyleNormal }; $p.Format.FirstLineIndent = 0; $p.Format.SpaceAfter = 6 }
            default { $p.Style = $wdStyleNormal; Apply-BodyFormat $p }
        }
        $idx++
    }
}

function Format-Table {
    param($table, [double[]]$Widths)
    $table.AllowAutoFit = $false
    $table.AutoFitBehavior($wdAutoFitWindow)
    $table.Rows.Item(1).HeadingFormat = -1
    $table.TopPadding = 5
    $table.BottomPadding = 5
    $table.LeftPadding = 6
    $table.RightPadding = 6
    $table.Borders.Enable = 1
    for ($c = 1; $c -le $table.Columns.Count; $c++) {
        if ($Widths -and $c -le $Widths.Count) {
            $table.Columns.Item($c).PreferredWidthType = $wdPreferredWidthPoints
            $table.Columns.Item($c).PreferredWidth = $Widths[$c-1]
        }
    }
    foreach ($row in $table.Rows) {
        foreach ($cell in $row.Cells) {
            $cell.VerticalAlignment = $wdCellAlignVerticalCenter
            foreach ($cp in $cell.Range.Paragraphs) {
                $cp.Style = $wdStyleNormal
                $cp.Format.FirstLineIndent = 0
            }
            $cell.Range.Font.Name = 'Times New Roman'
            $cell.Range.Font.NameFarEast = '宋体'
            $cell.Range.Font.Size = 10.5
            $cell.Range.ParagraphFormat.SpaceAfter = 0
            $cell.Range.ParagraphFormat.LineSpacingRule = 0
        }
    }
    foreach ($cell in $table.Rows.Item(1).Cells) {
        $cell.Range.Bold = -1
        $cell.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
        $cell.Shading.BackgroundPatternColor = 15790320
    }
}

function Insert-TableBeforeHeading {
    param($doc, [string]$TargetHeading, [string]$Lead, [string]$Caption, [array]$Rows, [double[]]$Widths)
    $target = Find-Paragraph $doc $TargetHeading
    $placeholder = '[[TABLE_' + [guid]::NewGuid().ToString('N') + ']]'
    Insert-BlocksBeforeHeading $doc $TargetHeading @(
        @{Text=$Lead; Style='Body'},
        @{Text=$Caption; Style='Caption'},
        @{Text=$placeholder; Style='NoIndent'}
    )
    $p = Find-Paragraph $doc $placeholder
    $pos = $p.Range.Start
    $p.Range.Delete()
    $r = $doc.Range($pos, $pos)
    $table = $doc.Tables.Add($r, $Rows.Count, $Rows[0].Count)
    for ($i = 0; $i -lt $Rows.Count; $i++) {
        for ($j = 0; $j -lt $Rows[$i].Count; $j++) {
            $table.Cell($i+1, $j+1).Range.Text = [string]$Rows[$i][$j]
        }
    }
    Format-Table $table $Widths
    $after = $doc.Range($table.Range.End, $table.Range.End)
    $after.InsertAfter("`r")
}

function Insert-FigureBeforeHeading {
    param($doc, [string]$TargetHeading, [string]$Path, [string]$Caption, [string]$Lead, [double]$MaxWidth = 440, [double]$MaxHeight = 340)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Figure missing: $Path" }
    $placeholder = '[[FIG_' + [guid]::NewGuid().ToString('N') + ']]'
    Insert-BlocksBeforeHeading $doc $TargetHeading @(
        @{Text=$Lead; Style='Body'},
        @{Text=$placeholder; Style='NoIndent'},
        @{Text=$Caption; Style='Caption'}
    )
    $p = Find-Paragraph $doc $placeholder
    $r = $p.Range.Duplicate
    $r.End = $r.End - 1
    $r.Text = ''
    $anchor = $doc.Range($p.Range.Start, $p.Range.Start)
    $pic = $doc.InlineShapes.AddPicture($Path, $false, $true, $anchor)
    $pic.LockAspectRatio = -1
    if ($pic.Width -gt $MaxWidth) { $pic.Width = $MaxWidth }
    if ($pic.Height -gt $MaxHeight) { $pic.Height = $MaxHeight }
    $p.Alignment = $wdAlignParagraphCenter
    $p.Format.FirstLineIndent = 0
    $p.Format.SpaceAfter = 3
    $p.Format.KeepWithNext = -1
    $captionP = Find-Paragraph $doc $Caption
    $captionP.Style = $wdStyleCaption
    $captionP.Alignment = $wdAlignParagraphCenter
    $captionP.Format.FirstLineIndent = 0
    $captionP.Format.SpaceAfter = 8
}

function Append-Paragraph {
    param($doc, [string]$Text, [string]$Style = 'Body')
    $pos = $doc.Content.End - 1
    $r = $doc.Range($pos, $pos)
    $r.InsertBefore($Text + "`r")
    $p = $doc.Range($pos, $pos + $Text.Length + 1).Paragraphs.Item(1)
    switch ($Style) {
        'H1' { $p.Style = $wdStyleHeading1 }
        'H2' { $p.Style = $wdStyleHeading2 }
        'H3' { $p.Style = $wdStyleHeading3 }
        'Caption' { $p.Style = $wdStyleCaption; $p.Alignment = $wdAlignParagraphCenter }
        'Code' { try { $p.Style = 'CodeBlock' } catch { $p.Style = $wdStyleNormal }; $p.Format.FirstLineIndent = 0; $p.Range.Font.Name = 'Consolas'; $p.Range.Font.NameFarEast = '等线'; $p.Range.Font.Size = 9 }
        'NoIndent' { try { $p.Style = 'BodyNoIndent' } catch { $p.Style = $wdStyleNormal }; $p.Format.FirstLineIndent = 0 }
        default { $p.Style = $wdStyleNormal; Apply-BodyFormat $p }
    }
    return $p
}

function Append-Table {
    param($doc, [string]$Caption, [array]$Rows, [double[]]$Widths)
    [void](Append-Paragraph $doc $Caption 'Caption')
    $pos = $doc.Content.End - 1
    $r = $doc.Range($pos, $pos)
    $table = $doc.Tables.Add($r, $Rows.Count, $Rows[0].Count)
    for ($i = 0; $i -lt $Rows.Count; $i++) {
        for ($j = 0; $j -lt $Rows[$i].Count; $j++) { $table.Cell($i+1, $j+1).Range.Text = [string]$Rows[$i][$j] }
    }
    Format-Table $table $Widths
    $after = $doc.Range($table.Range.End, $table.Range.End)
    $after.InsertAfter("`r")
}

function Append-Figure {
    param($doc, [string]$Path, [string]$Caption, [double]$MaxWidth = 440, [double]$MaxHeight = 340)
    $p = Append-Paragraph $doc '' 'NoIndent'
    $anchor = $doc.Range($p.Range.Start, $p.Range.Start)
    $pic = $doc.InlineShapes.AddPicture($Path, $false, $true, $anchor)
    $pic.LockAspectRatio = -1
    if ($pic.Width -gt $MaxWidth) { $pic.Width = $MaxWidth }
    if ($pic.Height -gt $MaxHeight) { $pic.Height = $MaxHeight }
    $p.Alignment = $wdAlignParagraphCenter
    $p.Format.KeepWithNext = -1
    [void](Append-Paragraph $doc $Caption 'Caption')
}

$word = New-Object -ComObject Word.Application
try {
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($final, $false, $false)

    # Base styles: preserve the existing visual language while normalizing body text and headings.
    $normal = $doc.Styles.Item($wdStyleNormal)
    $normal.Font.Name = 'Times New Roman'
    $normal.Font.NameFarEast = '宋体'
    $normal.Font.Size = 12
    $normal.ParagraphFormat.Alignment = $wdAlignParagraphJustify
    $normal.ParagraphFormat.LineSpacingRule = 1
    $normal.ParagraphFormat.SpaceAfter = 6
    $normal.ParagraphFormat.FirstLineIndent = 24

    foreach ($styleId in @($wdStyleHeading1, $wdStyleHeading2, $wdStyleHeading3)) {
        $s = $doc.Styles.Item($styleId)
        $s.Font.Name = 'Times New Roman'
        $s.Font.NameFarEast = '黑体'
        $s.Font.Color = 6108957
        $s.ParagraphFormat.KeepWithNext = -1
        $s.ParagraphFormat.KeepTogether = -1
    }
    $doc.Styles.Item($wdStyleHeading1).Font.Size = 16
    $doc.Styles.Item($wdStyleHeading1).Font.Bold = -1
    $doc.Styles.Item($wdStyleHeading1).ParagraphFormat.PageBreakBefore = -1
    $doc.Styles.Item($wdStyleHeading1).ParagraphFormat.SpaceBefore = 0
    $doc.Styles.Item($wdStyleHeading1).ParagraphFormat.SpaceAfter = 12
    $doc.Styles.Item($wdStyleHeading2).Font.Size = 14
    $doc.Styles.Item($wdStyleHeading2).Font.Bold = -1
    $doc.Styles.Item($wdStyleHeading2).ParagraphFormat.SpaceBefore = 12
    $doc.Styles.Item($wdStyleHeading2).ParagraphFormat.SpaceAfter = 6
    $doc.Styles.Item($wdStyleHeading3).Font.Size = 12
    $doc.Styles.Item($wdStyleHeading3).Font.Bold = -1
    $doc.Styles.Item($wdStyleHeading3).ParagraphFormat.SpaceBefore = 9
    $doc.Styles.Item($wdStyleHeading3).ParagraphFormat.SpaceAfter = 4
    $doc.Styles.Item($wdStyleCaption).Font.NameFarEast = '宋体'
    $doc.Styles.Item($wdStyleCaption).Font.Name = 'Times New Roman'
    $doc.Styles.Item($wdStyleCaption).Font.Size = 10.5
    $doc.Styles.Item($wdStyleCaption).Font.Bold = 0

    try { $front = $doc.Styles.Item('FrontHeading') } catch { $front = $doc.Styles.Add('FrontHeading', 1) }
    $front.Font.NameFarEast = '黑体'
    $front.Font.Name = 'Times New Roman'
    $front.Font.Size = 16
    $front.Font.Bold = -1
    $front.ParagraphFormat.Alignment = $wdAlignParagraphCenter
    $front.ParagraphFormat.SpaceBefore = 0
    $front.ParagraphFormat.SpaceAfter = 12
    $front.ParagraphFormat.OutlineLevel = 9

    # Remove the manually typed TOC block, retaining the original cover.
    $oldToc = Find-Paragraph $doc '目  录'
    $abstract = Find-Paragraph $doc '摘要'
    $doc.Range($oldToc.Range.Start, $abstract.Range.Start).Delete()
    $abstract = Find-Paragraph $doc '摘要'
    $abstract.Style = 'FrontHeading'
    $abstract.Format.PageBreakBefore = -1
    $keywords = Find-Paragraph $doc '关键词'
    $keywords.Style = 'FrontHeading'
    $keywords.Format.PageBreakBefore = 0
    $keywordBody = $keywords.Next()
    $tocPos = $keywordBody.Range.End
    $tocSeed = $doc.Range($tocPos, $tocPos)
    $tocSeed.InsertAfter(([char]12) + "目录`r[[AUTO_TOC]]`r")
    $tocHeading = Find-Paragraph $doc '目录'
    $tocHeading.Style = 'FrontHeading'
    $tocHeading.Format.PageBreakBefore = 0
    $tocPlaceholder = Find-Paragraph $doc '[[AUTO_TOC]]'
    $tocRange = $tocPlaceholder.Range.Duplicate
    $tocRange.End = $tocRange.End - 1
    $tocRange.Text = ''
    [void]$doc.TablesOfContents.Add($tocRange, $true, 1, 3, $true, '', $true, $true, '', $true, $false, $true)

    Replace-AllText $doc '电子工程相关专业' '电子工程'
    Replace-AllText $doc '二维 A* 导航' 'Motion-Aware A* 导航'
    Replace-AllText $doc '二维 A* 路径规划' 'Motion-Aware A* 路径规划'
    Replace-AllText $doc '2D A*' 'Motion-Aware A*'

    Set-ParagraphText (Find-Paragraph $doc '第三章 机器人比赛系统的重新适配与调试') '第三章 机器人比赛系统的重新适配与现场调试'
    Set-ParagraphText (Find-Paragraph $doc '3.1 新 TonyPi 平台与软件分层') '3.1 基于已有 robotall 的机器人软件分层'
    Set-ParagraphText (Find-Paragraph $doc '3.2 场地变化与地图重新建模') '3.2 场地变化与坐标重新测量'
    Set-ParagraphText (Find-Paragraph $doc '3.4 Motion-Aware A* 路径规划与运动执行') '3.4 地图与 Motion-Aware A* 路径规划'
    Set-ParagraphText (Find-Paragraph $doc '第八章 实习收获与能力提升') '第八章 实习成果与工程认识'
    Set-ParagraphText (Find-Paragraph $doc '7.4 总结出的排障方法') '7.4 多学生并发与调度问题'
    Set-ParagraphText (Find-Paragraph $doc '7.5 从临时修复到工程化改进') '7.5 问题排查方式的变化'

    # Chapter 1 additions.
    Insert-BlocksBeforeHeading $doc '1.2 实习目标' @(
        @{Text='在参加本次实习之前，我已经作为学生完成过“智能机器人设计实践”课程，并实际使用过当时的机器人比赛系统。因此，我对 AprilTag 定位、机器人导航、路径规划、FPGA 花卉识别和比赛流程已有基础认识。不过，当时的重点是按照既有接口完成实验；本次五周实习则需要站在维护者和建设者的角度，重新适配机器人、场地与计算平台，并将调试经验沉淀为可复现的部署方法和教学资料。'; Style='Body'},
        @{Text='实习开始时，系统的三项基础条件均发生了变化：旧机器人被新的 TonyPi 替代，比赛场地及 Tag/Screen 布局重新调整，FPGA 的使用方式也从客户端直连指定 KV260 演进为“Robot/Student—Central Server—KV260 Worker Pool—FPGA”。这些变化使原先隐含在代码中的坐标、动作、网络和资源假设必须被逐项重新验证。'; Style='Body'}
    )
    Insert-TableBeforeHeading $doc '1.3 主要技术内容' '三个目标既相互独立又逐步汇合，最终均以现场可验证的结果作为验收依据。' '表 1-1 实习目标、主要内容与最终成果' @(
        @('实习目标','主要内容','最终成果'),
        @('机器人系统重新适配','TonyPi、场地、定位、导航、识别、NFC','新场地完整比赛流程运行'),
        @('KV260 共享计算平台','SD 卡、Runtime、Worker、Central、调度、UI','七块 KV260 形成可调度 Worker'),
        @('教学资料更新','实验一至五、比赛说明、测试样例','完整新版指导书与验证流程')
    ) @(120,210,120)
    Insert-TableBeforeHeading $doc '第二章 总体方案与五周实施过程' '从技术层次看，实习工作覆盖感知、控制、FPGA Runtime、服务端和教学交付。表 1-2 概括了各模块在完整链路中的作用。' '表 1-2 主要技术模块及其作用' @(
        @('技术模块','关键技术','在实习中的作用'),
        @('机器人定位','AprilTag、PnP、坐标转换','获得可信世界坐标与朝向'),
        @('机器人导航','地图、Motion-Aware A*、动作执行','规划并执行可落地的离散动作'),
        @('机器人交互','FPGA 花卉分类、NFC','判断并完成换花任务'),
        @('KV260 Runtime','XRT、ZOCL、pyxrt、PYNQ','建立 FPGA 运行环境'),
        @('Central Server','FastAPI、SQLite、认证','统一管理请求和持久化状态'),
        @('调度与管理','Worker、Lease、FIFO、Dashboard','多学生共享多板并支持排障')
    ) @(115,210,125)

    # Chapter 2 additions and diagram.
    Insert-FigureBeforeHeading $doc '2.2 五周工作安排与演进' (Join-Path $work 'diagram_system.svg') '图 2-1 机器人任务系统与 KV260 共享平台总体架构' '系统由 Robot Mission System 与 KV260 Shared FPGA Platform 两个子系统组成，并在第五章完成端到端集成，如图 2-1 所示。' 450 300
    Insert-TableBeforeHeading $doc '2.3 开发与验证方法' '时间安排遵循“先单模块稳定，再逐步集成”的路线。第一周先建立机器人现场基线，之后再由单板扩展到中央调度，避免多个不确定因素同时叠加。' '表 2-2 五周实施过程与关键成果' @(
        @('周次','工作重点','关键成果'),
        @('第一周','新机器人与新场地调试','完整比赛流程现场跑通'),
        @('第二周','KV260 镜像及 Runtime 环境','第一块 KV260 成为可用 Worker'),
        @('第三周','Central Server 与双板调度','多 Worker 基础调度跑通'),
        @('第四周','多板部署、持久化与 UI','多学生、多 Worker 平台完善'),
        @('第五周','机器人接入与指导书更新','端到端联调和文档交付完成')
    ) @(70,230,150)

    # Chapter 3: provenance, geometry, current planner and field evidence.
    Insert-BlocksBeforeHeading $doc '3.2 场地变化与坐标重新测量' @(
        @{Text='需要特别说明的是，robotall 并非我在这五周内从零开发的模块，而是此前已经由其他人员完善并提供的底层机器人功能层。本次工作的重点是在这一基础上重新梳理并完善 robot_tonypi，将定位、地图、Motion-Aware A*、任务状态、花卉分类、NFC 交互和 Debug 逻辑集中在上层，使硬件能力与比赛策略保持清晰边界。'; Style='Body'}
    )
    Insert-FigureBeforeHeading $doc '3.2 场地变化与坐标重新测量' (Join-Path $work 'diagram_robot_layers.svg') '图 3-1 基于已有 robotall 的 TonyPi 软件分层' '图 3-1 展示了本次实习实际调整的上层范围，以及它与已有 robotall 和原厂动作系统之间的关系。' 420 300
    Insert-FigureBeforeHeading $doc '3.2 场地变化与坐标重新测量' (Join-Path $images '01_TonyPi机器人硬件调试.jpg') '图 3-2 新 TonyPi 机器人硬件与接线调试' '在软件适配前，首先检查了新 TonyPi 的控制板、舵机、相机和接线状态，如图 3-2 所示。' 420 310
    Insert-BlocksBeforeHeading $doc '3.3 AprilTag 定位与位姿质量控制' @(
        @{Text='场地调整后，我重新逐个测量可用 AprilTag 的位置，并同步更新 load_pos.py 与 competition_config.json 中的场地、建筑、障碍和目标信息。测量工作的关键不是单独记录某个 Tag 坐标，而是确保定位、地图、规划和目标交互对同一物理位置采用一致约定。'; Style='Body'},
        @{Text='与旧场地相比，Tag 与对应 Screen 的相对放置由 Screen 右上角调整为左上角。这一变化直接影响 Screen 中心推算、Tag 到目标面的几何关系、机器人最终交互点和期望朝向。当前视觉逻辑按“左上 Tag”建立途中几何绑定，因此旧坐标不能直接沿用，必须在现场重新测量并结合画面验证。'; Style='Body'}
    )
    Insert-FigureBeforeHeading $doc '3.3 AprilTag 定位与位姿质量控制' (Join-Path $work 'diagram_tag_layout.svg') '图 3-3 Tag 与 Screen 相对布局变化示意' '旧、新布局的差异及其对几何推算的影响如图 3-3 所示。' 440 230
    Insert-FigureBeforeHeading $doc '3.3 AprilTag 定位与位姿质量控制' (Join-Path $images '02_修改后比赛场地_01.jpg') '图 3-4 修改后的比赛场地及 AprilTag 布局' '完成坐标更新后，通过场地全景和 Debug 地图交叉检查建筑、道路、地面 Tag 与机器人位置，现场布局如图 3-4 所示。' 440 310
    $plannerP = Find-Paragraph $doc '机器人得到当前位置和目标交互点后，采用二维八邻域 A* 在栅格地图上计算可行路径。A* 使用欧氏距离作为启发函数，并结合障碍膨胀和代价图避免机器人过于贴近墙面。规划得到的原始栅格路径通常包含较多折点，因此进一步通过直线可通行性检查进行路径平滑，尽量删除不必要的中间网格点，使机器人在真实场地中的动作更直接。'
    Set-ParagraphText $plannerP '机器人得到当前位置和目标交互点后，当前正式版本采用 Motion-Aware A*。规划状态同时包含离散位置与物理 yaw，并直接搜索 FORWARD、STRAFE、TURN 和受限 REVERSE 等可执行动作，而不是先生成理想几何路径后再由执行器重新猜测动作。场地按 5 cm 分辨率建立栅格，结合障碍膨胀、通行代价、转向代价、动作切换代价和墙面净距约束，减少双足机器人在狭窄区域频繁转向或贴墙行走。'
    Insert-BlocksBeforeHeading $doc '3.5 花卉识别与目标确认' @(
        @{Text='动作执行必须使用实测模型。现场测试发现，左右转动作即使调用次数相同，实际角度也可能不同；前进、侧移和后退的真实位移同样会受地面摩擦、电量与姿态影响。连续动作会累积误差，因此 dead reckoning 只作为短期估计，达到动作预算、接近障碍或接近目标时强制重新进行视觉定位。当前配置分别保存左右动作参数，使 Planner 的预测与 Executor 的实机行为保持一致。'; Style='Body'}
    )
    Insert-FigureBeforeHeading $doc '第四章 KV260 FPGA 共享计算平台的设计与实现' (Join-Path $images '03_机器人现场完整运行_01.jpg') '图 3-5 TonyPi 在新场地完成现场运行验证' '定位、规划、分类与交互逻辑完成组合后，在真实场地按完整任务流程进行验证，如图 3-5 所示。' 330 390

    # Chapter 4 tables, diagrams and evidence.
    Insert-TableBeforeHeading $doc '4.2 SD 卡镜像与 KV260 标准化部署' '单板直连能够证明 FPGA 分类链路可行，但不能直接扩展为面向多学生的教学平台。主要差距归纳见表 4-1。' '表 4-1 单板直连模式的问题与多板平台需求' @(
        @('问题','单板/直连模式表现','多板平台需求'),
        @('计算资源固定','用户绑定某块 KV260','Central 统一选择 Worker'),
        @('FPGA 设计不同','bit/hwh 容易互相覆盖','Artifact 版本化保存'),
        @('多用户竞争','资源冲突或直接失败','Queue + Lease + Scheduler'),
        @('状态不可追踪','难以判断任务运行位置','Request 持久化 + UI + Audit')
    ) @(95,170,185)
    Insert-TableBeforeHeading $doc '4.2 SD 卡镜像与 KV260 标准化部署' '围绕上述问题，第四章的工程实现被拆分为环境、节点、入口、数据、调度、恢复、可观测性和验证等模块。' '表 4-2 第四章平台设计模块总览' @(
        @('小节','模块','解决的问题'),
        @('4.2','SD 卡镜像与批量部署','多板环境一致性'),
        @('4.3','XRT/ZOCL/pyxrt/PYNQ Runtime','FPGA 运行环境'),
        @('4.4','Central Server 与 Worker 职责','统一入口和执行边界'),
        @('4.5','Student、Artifact 与认证','学生设计管理'),
        @('4.6～4.7','Request、Lease、Scheduler、Queue','多用户多板调度'),
        @('4.8～4.9','恢复、Audit 与管理 UI','可靠性和可观测性'),
        @('4.10','多板与并发测试','系统验证')
    ) @(65,205,180)
    Insert-FigureBeforeHeading $doc '4.3 PYNQ、Overlay 与 FPGA 数据通路' (Join-Path $images '04_SD卡读卡器异常日志.png') '图 4-1 SD 卡写入过程中的 I/O 与越界异常' '批量写卡阶段曾出现 write fault、I/O error 和 access beyond end of device，现场证据如图 4-1 所示。排查结果说明部分故障来自 SD 卡或读卡器等介质链路，而非脚本逻辑本身。' 445 300
    Insert-FigureBeforeHeading $doc '4.3 PYNQ、Overlay 与 FPGA 数据通路' (Join-Path $images '05_KV260首次启动_成功.jpg') '图 4-2 KV260 完成首次启动并加载系统服务' '在重新写卡并检查介质、分区和首次启动配置后，KV260 能够正常进入系统并启动相关服务，如图 4-2 所示。' 430 305
    Insert-FigureBeforeHeading $doc '4.5 学生认证与 FPGA Artifact 管理' (Join-Path $work 'diagram_central.svg') '图 4-3 Central Server 与 KV260 Worker 的职责边界' 'Central 负责身份、状态与资源所有权，Worker 专注于 Overlay、DMA 和硬件执行，职责关系如图 4-3 所示。' 445 295
    Insert-FigureBeforeHeading $doc '4.8 Worker Registry、健康检查与故障恢复' (Join-Path $work 'diagram_scheduler.svg') '图 4-4 Request、Lease 与 Worker 调度流程' 'Request 先持久化，再依据可用资源进入同步完成或异步排队路径，完整流程如图 4-4 所示。' 360 360
    Insert-FigureBeforeHeading $doc '4.10 并发、持久化与恢复测试' (Join-Path $images '14_Central_Server平台总览.png') '图 4-5 Central Server 平台总览与 Worker 状态' 'Dashboard 将 Worker、Request 与利用率集中展示，为现场调试提供统一入口，如图 4-5 所示。' 445 275
    Insert-FigureBeforeHeading $doc '4.10 并发、持久化与恢复测试' (Join-Path $images '12_Request查询页面.png') '图 4-6 按 Request ID 查询计算状态' '针对“后台已完成但前端不易定位”的问题，管理界面增加了 Request ID 查询，如图 4-6 所示。' 445 275
    Insert-FigureBeforeHeading $doc '4.10 并发、持久化与恢复测试' (Join-Path $images '19_Artifact管理页面.png') '图 4-7 Artifact 上传、版本与归档管理' 'Artifact 页面同时承担 bit/hwh 上传、版本查看、哈希核对和安全清理预览，见图 4-7。' 445 275
    Insert-BlocksBeforeHeading $doc '4.11 技术选择与工程取舍' @(
        @{Text='多板验证按照“单块成功—第二块加入—批量烧卡—七块在线—并发请求”逐步推进。第一块板证明 Runtime 与 Worker 链路可行，第二块板用于验证分配和释放不再依赖单节点假设，随后才扩大到七块板卡。最后使用五名虚拟 Student 同时上传 Artifact 和提交 predict，请求既出现可立即完成的 HTTP 200，也出现持久化排队的 HTTP 202；通过 request_id 持续查询后，五个请求全部进入 COMPLETED，且未出现 Worker ownership 冲突。'; Style='Body'}
    )
    Insert-FigureBeforeHeading $doc '4.11 技术选择与工程取舍' (Join-Path $images '07_七台KV260批量部署_01.jpg') '图 4-8 七台 KV260 的集中接线与批量部署' '最终完成七块 KV260 的标准化部署，实物连接如图 4-8 所示。单板成功证明方案可行，七板上线则进一步证明部署流程可复制。' 430 300
    Insert-FigureBeforeHeading $doc '4.11 技术选择与工程取舍' (Join-Path $images '09_多学生并发调度测试_03.png') '图 4-9 五名 Student 并发请求全部完成' '五 Student 并发与持久化测试的最终结果如图 4-9 所示，Completed 为 5、Failed 为 0。' 445 235
    Insert-FigureBeforeHeading $doc '4.11 技术选择与工程取舍' (Join-Path $images '20_七台KV260在线调度_01.png') '图 4-10 七台 KV260 在线调度状态' '完成批量部署后，Central Server 能在同一界面识别并管理七台在线 Worker，见图 4-10。' 445 270

    # Chapter 5: preserve history and add exact direct endpoint evidence.
    $directP = Find-Paragraph $doc '机器人最初采用 direct 模式，classifier-url 直接指向某一块 KV260 Worker 的 /predict。这一方式适合独立测试，但机器人需要知道具体 Worker IP，并且无法判断该板是否正被其他学生占用。共享平台完成后，机器人增加 central 模式，将 classifier-url 改为 Central Server 的 /predict，并携带 classifier-student-id 和密码。'
    Set-ParagraphText $directP '机器人最初采用 direct 模式，classifier-url 直接配置为 http://192.168.31.81:8080/predict。TonyPi 将裁剪后的花卉图像以 multipart/form-data 的 image 字段发送给该 KV260 上的 fpga_flower_server/fpga_server_api_ready.py；Flask 服务在 8080 端口串行接收图像，加载 design_1_wrapper.bit/hwh，通过 PYNQ Overlay 与 axi_dma_0 执行 12 类花卉推理并返回结果。这一方式适合单板链路验证，但机器人与具体 IP 强绑定，板卡离线后必须人工修改地址，也无法统一处理多学生竞争。'
    Insert-FigureBeforeHeading $doc '5.2 完整数据链路' (Join-Path $work 'diagram_evolution.svg') '图 5-1 机器人从直连固定 KV260 演进为 Central 模式' '早期直连和当前共享模式的差异如图 5-1 所示。接入 Central 后，Robot 不再知道具体 KV260 地址。' 445 270
    Insert-FigureBeforeHeading $doc '5.4 系统集成带来的改进' (Join-Path $images '08_HTTP_FPGA分类请求测试.png') '图 5-2 Central Server 返回 completed 与 queued 请求' '端到端 HTTP 测试同时观察到立即完成与进入队列两种合法结果，如图 5-2 所示；queued 请求由 request_id 延续查询，而不是重复提交。' 445 230

    # Chapter 6: exact manual contents from current Markdown files.
    Insert-TableBeforeHeading $doc '6.3 比赛说明与 NFC/服务器通信更新' '我逐份打开并核对了实验一至实验五的当前 Markdown 版本，使修改方向与实际教学内容对应，而不是按文件名推测。' '表 6-1 实验一至实验五的主要更新方向' @(
        @('指导书','当前主题','主要修改方向'),
        @('实验一','开机、基础操作与控制','TonyPi 环境、启动/关机、SSH/SCP、robotall、ActionGroup 与相机'),
        @('实验二','AprilTag 识别','tag36h11、检测结果、图像处理与识别接口'),
        @('实验三','定位','相机标定、PnP、坐标变换、load_pos.py 世界坐标'),
        @('实验四','路径规划','新场地地图、定位闭环、动作误差、规划与重定位'),
        @('实验五','神经网络加速器硬件实践','bit/hwh、KV260、Central HTTP、200/202 与 request_id')
    ) @(70,130,250)
    $sixTwo = Find-Paragraph $doc '实验一主要更新新 TonyPi 的硬件与软件环境、内部目录、启动和关机方式，使学生能够正确进入机器人开发环境。后续实验根据新场地重新整理 AprilTag、地图建模、定位和导航相关内容，并提供与当前代码一致的示例。涉及 FPGA 的实验则更新 KV260/PYNQ 的运行方式、bit/hwh 使用以及 HTTP 通信说明。'
    Set-ParagraphText $sixTwo '实验一更新 TonyPi 2025 环境、开关机、文件传输、robotall 安装、相机与 ActionGroup 基础动作；实验二围绕 tag36h11 AprilTag 的线段、四边形、单应性和检测库接口；实验三补充相机标定、PnP、机器人方位与 load_pos.py 中的世界坐标；实验四结合新场地说明地图建模、动作控制、A* 等规划方法和周期重定位闭环；实验五保留 CNN 训练与 bitstream 生成，同时将 FPGA 调用方式更新为 Artifact 上传、Central Server :8000、KV260 Worker :8080、HTTP 200/202 和 request_id 查询。'
    Insert-BlocksBeforeHeading $doc '6.4 文档工作的工程价值' @(
        @{Text='教学资料的三类核心变化可以归纳为：第一，机器人平台由旧设备切换为 TonyPi，目录、动作组、相机和运行方法需要统一；第二，真实换花链路改为“final forward—举左手—NFC”，失败后按照有限重试与恢复语义处理；第三，FPGA 分类由客户端直连指定板卡转为通过 Central Server 上传 Artifact 并提交 predict。指导书修改后，又用 demo 与测试入口分别检查 NFC、HTTP、FPGA 和 Robot 流程，确保学生按文档能够得到可验收结果。'; Style='Body'}
    )

    # Chapter 7: detailed real problems and bounded solutions.
    Insert-BlocksBeforeHeading $doc '7.2 机器人定位与运动误差' @(
        @{Text='7.1.1 初期 Runtime、XRT、ZOCL 与 PYNQ 环境'; Style='H3'},
        @{Text='KV260 的 FPGA Python 环境不是安装单个包即可完成。Ubuntu、FPGA Manager、XRT userspace、与内核匹配的 ZOCL、pyxrt、Minimal PYNQ、设备树和 systemd 服务必须形成一致链路。初期 ZOCL 和驱动环境调试耗时较长，因此后续把版本检查、模块加载、PYNQ ON_TARGET、XRT 枚举和 Overlay 前置检查写入 runtime_init_kv260.sh 与配套检查脚本。'; Style='Body'},
        @{Text='7.1.2 SD 卡与读卡器 I/O 问题'; Style='H3'},
        @{Text='实际写卡过程中先后看到 I/O error、access beyond end of device、Medium Error、Unrecovered read error 和“设备上无剩余空间”等现象。排查没有直接认定为脚本缺陷，而是依次比较镜像与设备容量、重新识别整盘设备、检查内核日志，并更换 SD 卡和读卡器。结果表明部分问题来自存储介质和读卡链路，说明软件报错不一定等于软件逻辑错误。'; Style='Body'},
        @{Text='7.1.3 SSH、网络与首次启动问题'; Style='H3'},
        @{Text='写卡成功后仍遇到 kex_exchange_identification reset、Connection refused 和 ping 失败。排查按“板卡是否完成启动—网卡名称与 netplan 是否匹配—静态 IP 与路由是否生效—cloud-init 是否完成—SSH socket/service 是否监听”的顺序逐层进行，避免仅凭一个 SSH 现象重烧整张卡。'; Style='Body'},
        @{Text='7.1.4 烧卡时间与重复失败'; Style='H3'},
        @{Text='单张卡写入、扩容、首次启动和 Runtime 安装耗时较长，第一张成功也不意味着下一张一定成功。批量过程中任何介质不稳定、设备名变化或网络差异都会放大时间成本。因此每张卡都重新核对目标设备、Board ID、hostname 和 IP，并保留阶段性验证点，使失败后能够从明确阶段恢复。'; Style='Body'},
        @{Text='7.1.5 从一块成功到七块上线'; Style='H3'},
        @{Text='第一块 KV260 成功证明镜像、Runtime、Worker 与 FPGA 路径可行；第二块加入证明 Central 调度不再依赖单节点；最终七块板卡完成批量部署并同时在线，才说明方法具有可复制性。这个过程也促使脚本将人工经验固化为 Board ID、hostname、静态网络和运行环境的标准化规则。'; Style='Body'}
    )
    Insert-BlocksBeforeHeading $doc '7.3 网络与服务器状态不一致问题' @(
        @{Text='7.2.1 动作误差与累计偏移'; Style='H3'},
        @{Text='左右转的实际角度不完全对称，同一动作的位移也会随电量、地面和姿态变化。解决方法不是只修改 Planner，而是先测量单动作，再校准配置，并把视觉重定位作为闭环反馈；dead reckoning 仅在短时间内辅助估计。'; Style='Body'},
        @{Text='7.2.2 定位跳变与错误候选'; Style='H3'},
        @{Text='相机边缘、小面积 Tag、多 Tag 候选和动作后的模糊画面可能导致 Pose 突然漂移。当前代码结合最小面积、边缘过滤、场地与建筑约束、pose conflict、hard jump 和 suspect confirmation 处理候选，并通过 Debug 地图检查拒绝原因。'; Style='Body'},
        @{Text='7.2.3 NFC 换花失败恢复'; Style='H3'},
        @{Text='NFC 并非每次一次成功。Attempt 1 失败后，机器人先 retreat，再 relocate 并重新获取当前目标；重新通过 FPGA 判断花卉状态后，才决定是否进行 Attempt 2。若在有限轮次内无法确认当前目标，则进入 GAVE_UP 并继续任务，避免局部交互失败拖垮整场流程。'; Style='Body'},
        @{Text='7.2.4 恢复路径中的死循环'; Style='H3'},
        @{Text='recovery、reacquire 和 navigation failure 若没有次数上限，容易在物理环境不满足时反复执行。当前状态机通过目标重获上限、NFC Attempt 上限、navigation recovery 上限、请求 deadline 和 GAVE_UP 终态约束循环，使失败路径同样具有可测试的结束条件。'; Style='Body'}
    )
    $sevenFourBody = (Find-Paragraph $doc '通过上述问题，实习中逐渐形成较稳定的排障方法：第一，先记录准确现象和可复现条件；第二，按系统层次拆分链路，例如 Robot、Network、Central、Worker、PYNQ、FPGA；第三，在每一层设置最小独立测试，如 curl health、单 Worker predict、数据库 Request 查询或 dry-run；第四，一次只修改有限变量并保留日志；第五，问题解决后把检查方法写入脚本、Dashboard 或指导书，避免同类问题依赖个人记忆再次排查。')
    Set-ParagraphText $sevenFourBody '多学生并发的核心问题不是请求能否到达，而是资源所有权在并发下是否仍唯一。测试中同时检查同一 Student 最多一个 Lease、单 Worker 最多一个 owner、同一 Student 请求 FIFO、BUSY 不回收和无资源时先持久化再返回 202。收到 202 后使用同一 request_id 查询，Worker 释放后由 allocator 继续执行；服务器重启后，Request、Lease、Artifact 和 Worker 视图仍可从 SQLite 恢复。'
    $sevenFiveBody = Find-Paragraph $doc '实习中不少问题最初都可以通过临时命令绕过，例如手工修改某块板的网络、直接重启某个服务、在机器人代码中写死一个 Worker IP，或者在发生异常后由管理员手工删除状态。但如果只满足于临时恢复，随着设备和学生数量增加，同一个问题会不断重复出现。因此后期解决问题时尽量追问：这个故障能否通过脚本提前检测，能否通过状态机自动恢复，能否通过 UI 更快观察，能否通过指导书让其他人自行排查。'
    Set-ParagraphText $sevenFiveBody '随着系统规模扩大，排障方法逐步由“看到错误后尝试修改”转变为结构化流程：先记录现象和复现条件，再判断故障位于 Robot、Network、Central、Worker、PYNQ 还是 FPGA；随后使用最小化测试和日志验证假设，一次只改变有限变量，修复后重新执行端到端回归。最终还要把检查固化到脚本、状态机、Dashboard 或指导书中，使同类问题不再依赖个人记忆。'

    # Chapter 8 result table.
    Insert-TableBeforeHeading $doc '8.1 从算法实现到系统工程' '本次实习的成果不仅是若干功能点，而是从机器人现场任务、共享 FPGA 计算到教学资料的一条完整交付链。' '表 8-1 实习主要成果' @(
        @('成果类别','最终成果'),
        @('Robot','新 TonyPi 与新场地完整比赛流程'),
        @('KV260 Runtime','标准化镜像、XRT/ZOCL/pyxrt/PYNQ 环境'),
        @('Server','多学生、多 Worker、持久化调度平台与 UI'),
        @('Deployment','七块 KV260 在线可调度'),
        @('Integration','Robot → Central → KV260 → FPGA 端到端链路'),
        @('Documentation','实验一至五与比赛说明完成更新')
    ) @(130,320)

    # Replace old appendices with the required Appendix A.
    $appendixA = Find-Paragraph $doc '附录 A 主要仓库与工程成果'
    $doc.Range($appendixA.Range.Start, $doc.Content.End - 1).Delete()
    [void](Append-Paragraph $doc '附录A 项目代码与系统结构' 'H1')
    [void](Append-Paragraph $doc 'A.1 GitHub 仓库' 'H2')
    [void](Append-Paragraph $doc '本次实习的代码、文档与报告由主仓库组织，KV260 共享计算平台作为独立子模块维护。两个仓库分别承担总体集成与 FPGA 平台实现。' 'Body')
    Append-Table $doc '表 A-1 项目主要仓库' @(
        @('仓库','地址','主要内容'),
        @('Intern','https://github.com/shenghong1130/Intern','robot code、Laboratory Manual、Intern report 与 FPGA submodule'),
        @('FPGA_KV260','https://github.com/shenghong1130/FPGA_KV260','部署、Runtime、Worker、Central Server、测试、UI 与架构文档')
    ) @(80,190,180)
    foreach ($url in @('https://github.com/shenghong1130/Intern','https://github.com/shenghong1130/FPGA_KV260')) {
        $r = $doc.Content.Duplicate
        $find = $r.Find
        $find.Text = $url
        if ($find.Execute()) { [void]$doc.Hyperlinks.Add($r, $url) }
    }
    [void](Append-Paragraph $doc 'A.2 项目主要文件结构' 'H2')
    [void](Append-Paragraph $doc '以下结构从当前本地目录提炼，只保留与本报告直接相关的核心模块。' 'Body')
    [void](Append-Paragraph $doc @'
Intern/
├─ 0_FPGA_code/                  # KV260 共享 FPGA 平台
│  ├─ KV260_PYNQ_Framework.md
│  ├─ KV260_PYNQ_Architecture_Notes.md
│  ├─ KV260_SD_Card_Setup_Guide.md
│  ├─ KV260_Server_Usage_Guide.md
│  ├─ prepare_kv260_image.sh
│  ├─ runtime_init_kv260.sh
│  ├─ runtime/
│  ├─ worker/
│  └─ server/
│     ├─ app/
│     ├─ config/
│     ├─ testbed/
│     ├─ tests/
│     └─ ui/
├─ 1_robot_code/
│  ├─ robotall/
│  ├─ robot_tonypi/
│  │  ├─ main.py / config.py / load_pos.py
│  │  ├─ localizer.py / map_model.py / motion.py
│  │  ├─ classifier.py / interaction_logic.py / debug.py
│  │  └─ tests/
│  ├─ test/
│  ├─ doc.md
│  └─ 智能机器人及场地架构说明.md
├─ 2_Laboratory Manual/
│  ├─ 实验一指导书.md … 实验五指导书.md
│  └─ 机器人比赛说明.md
└─ 3_Intern report/
   ├─ images/
   └─ 实习报告_机器人与KV260共享计算平台_许胜宏_最终版.docx
'@ 'Code')
    [void](Append-Paragraph $doc 'A.3 项目总体架构图' 'H2')
    [void](Append-Paragraph $doc '主仓库中的机器人任务系统、KV260 共享平台和教学资料以 HTTP 集成链路为中心相互关联，整体结构如图 A-1 所示。' 'Body')
    Append-Figure $doc (Join-Path $work 'diagram_appendix.svg') '图 A-1 Intern 项目总体结构' 440 300

    # Normalize body paragraphs without disturbing cover/table paragraphs.
    foreach ($p in $doc.Paragraphs) {
        $styleName = [string]$p.Style
        $t = $p.Range.Text.Trim([char]13, [char]7, [char]32, [char]9)
        if ($t -and $styleName -match 'Normal|正文|Body') { Apply-BodyFormat $p }
    }
    foreach ($p in $doc.Paragraphs) {
        $t = $p.Range.Text.Trim([char]13, [char]7, [char]32, [char]9)
        if ($t -match '^(第一章|第二章|第三章|第四章|第五章|第六章|第七章|第八章|第九章|附录A)') {
            $p.Style = $wdStyleHeading1
            $p.Format.PageBreakBefore = -1
        }
    }
    foreach ($frontText in @('摘要','关键词','目录')) {
        $fp = Find-Paragraph $doc $frontText
        $fp.Style = 'FrontHeading'
        $fp.Format.OutlineLevel = 9
    }

    # Footer page number fields and field-update-on-open.
    foreach ($section in $doc.Sections) {
        $footer = $section.Footers.Item(1)
        $footer.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
        if ($footer.Range.Fields.Count -eq 0) {
            $fr = $footer.Range.Duplicate
            $fr.Collapse($wdCollapseEnd)
            [void]$footer.Range.Fields.Add($fr, 33)
        }
    }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Save()
    $doc.Close($true)

    # Reopen once to ensure the package is valid and TOC/page fields are refreshed in Word.
    $doc = $word.Documents.Open($final, $false, $false)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Save()
    Write-Output ('FINAL=' + $final)
    Write-Output ('PAGES=' + $doc.ComputeStatistics(2))
    Write-Output ('WORDS=' + $doc.ComputeStatistics(0))
    Write-Output ('PARAGRAPHS=' + $doc.Paragraphs.Count)
    Write-Output ('TABLES=' + $doc.Tables.Count)
    Write-Output ('INLINE_SHAPES=' + $doc.InlineShapes.Count)
    Write-Output ('TOCS=' + $doc.TablesOfContents.Count)
    $doc.Close($true)
}
finally {
    if ($word) { $word.Quit() }
}



