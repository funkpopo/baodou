param(
    [string]$OutputDir = (Join-Path $PSScriptRoot '..\benchmarks\cases\artifacts')
)

Set-StrictMode -Version Latest
Add-Type -AssemblyName System.Drawing

$resolvedRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDir)
$expectedRoot = [System.IO.Path]::GetFullPath((Join-Path $resolvedRoot 'benchmarks\cases'))
if (-not $resolvedOutput.StartsWith($expectedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must stay under $expectedRoot"
}
[System.IO.Directory]::CreateDirectory($resolvedOutput) | Out-Null

$fontName = 'Microsoft YaHei UI'
function New-Font([float]$size, [System.Drawing.FontStyle]$style = [System.Drawing.FontStyle]::Regular) {
    [System.Drawing.Font]::new($fontName, $size, $style, [System.Drawing.GraphicsUnit]::Pixel)
}
function New-Brush([string]$hex) {
    [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($hex))
}
function Paint-Text($g, [string]$text, [float]$x, [float]$y, [float]$size = 22, [string]$color = '#182230', [bool]$bold = $false) {
    $style = if ($bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    $font = New-Font $size $style
    $brush = New-Brush $color
    try { $g.DrawString($text, $font, $brush, $x, $y) } finally { $font.Dispose(); $brush.Dispose() }
}
function Paint-Rect($g, [string]$color, [float]$x, [float]$y, [float]$w, [float]$h) {
    $brush = New-Brush $color
    try { $g.FillRectangle($brush, $x, $y, $w, $h) } finally { $brush.Dispose() }
}
function Paint-Line($g, [string]$color, [float]$width, [float]$x1, [float]$y1, [float]$x2, [float]$y2) {
    $pen = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml($color), $width)
    try { $g.DrawLine($pen, $x1, $y1, $x2, $y2) } finally { $pen.Dispose() }
}
function Paint-Window($g, [string]$title, [string]$body = '#F7F9FC', [float]$x = 55, [float]$y = 42, [float]$w = 1170, [float]$h = 630) {
    Paint-Rect $g '#D7DEE8' ($x + 8) ($y + 10) $w $h
    Paint-Rect $g $body $x $y $w $h
    Paint-Rect $g '#E8EDF4' $x $y $w 52
    Paint-Text $g $title ($x + 20) ($y + 13) 22 '#182230' $true
    Paint-Text $g '—   □   ×' ($x + $w - 115) ($y + 13) 20 '#4D5968'
}
function Save-Fixture([string]$name, [scriptblock]$draw) {
    $bitmap = [System.Drawing.Bitmap]::new(1280, 720, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $g = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
        $g.Clear([System.Drawing.ColorTranslator]::FromHtml('#EEF2F7'))
        & $draw $g
        $path = Join-Path $resolvedOutput $name
        $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
        Write-Host "wrote $path"
    } finally {
        $g.Dispose()
        $bitmap.Dispose()
    }
}

Save-Fixture 'chat-email-inbox.png' {
    param($g)
    Paint-Window $g '示例邮箱 — 收件箱'
    Paint-Rect $g '#F0F4F9' 55 94 245 578
    Paint-Text $g '收件箱  3' 88 130 26 '#2457C5' $true
    Paint-Text $g '已加星标' 88 186 21
    Paint-Text $g '已发送' 88 232 21
    Paint-Text $g '主题' 334 118 18 '#667085' $true
    Paint-Text $g '测试构建已完成' 334 165 24 '#182230' $true
    Paint-Text $g '自动化服务 · 10:24' 880 169 18 '#667085'
    Paint-Line $g '#D8DEE8' 1 315 207 1196 207
    Paint-Text $g '周会纪要（示例数据）' 334 233 23
    Paint-Text $g '项目演示组 · 昨天' 895 237 18 '#667085'
    Paint-Line $g '#D8DEE8' 1 315 278 1196 278
    Paint-Text $g '欢迎使用脱敏测试邮箱' 334 304 23
}

Save-Fixture 'browser-home.png' {
    param($g)
    Paint-Window $g '浏览器 — 示例知识库'
    Paint-Rect $g '#FFFFFF' 86 111 1108 48
    Paint-Text $g 'https://docs.example.test/start' 112 121 19 '#536172'
    Paint-Text $g '示例知识库' 112 226 44 '#173A70' $true
    Paint-Text $g '本地视觉助手测试页面' 112 294 27 '#44546A'
    Paint-Rect $g '#E8F0FE' 112 360 310 132
    Paint-Text $g '快速开始' 142 386 26 '#2457C5' $true
    Paint-Text $g '查看离线使用说明' 142 435 20 '#334155'
    Paint-Rect $g '#EDF8F1' 452 360 310 132
    Paint-Text $g '运行状态' 482 386 26 '#18794E' $true
    Paint-Text $g '服务正常' 482 435 20 '#334155'
}

Save-Fixture 'ide-terminal-error.png' {
    param($g)
    Paint-Window $g '示例工程 — 编辑器' '#1E1E1E'
    Paint-Rect $g '#252526' 55 94 255 578
    Paint-Text $g 'EXPLORER' 78 118 16 '#C8C8C8' $true
    Paint-Text $g 'src' 84 160 18 '#D4D4D4'
    Paint-Text $g '  main.rs' 84 198 18 '#D4D4D4'
    Paint-Text $g 'TERMINAL' 340 405 16 '#C8C8C8' $true
    Paint-Line $g '#444444' 1 310 388 1225 388
    Paint-Text $g '> cargo test' 342 446 20 '#D4D4D4'
    Paint-Text $g 'error[E0425]: cannot find value `fixture_id` in this scope' 342 489 19 '#F48771' $true
    Paint-Text $g ' --> src/main.rs:42:17' 342 529 18 '#9CDCFE'
    Paint-Text $g 'test result: FAILED. 8 passed; 1 failed' 342 588 19 '#F48771'
}

Save-Fixture 'spreadsheet-numbers.png' {
    param($g)
    Paint-Window $g '示例季度统计表'
    $x = 94; $y = 135; $cellW = 210; $cellH = 64
    $headers = @('项目', '一月', '二月', '三月', '合计')
    for ($i = 0; $i -lt 5; $i++) { Paint-Rect $g '#DCE8F8' ($x + $i*$cellW) $y $cellW $cellH; Paint-Text $g $headers[$i] ($x + $i*$cellW + 18) ($y + 17) 20 '#1F3B64' $true }
    $rows = @(@('Alpha','128','145','162','435'), @('Beta','96','104','119','319'), @('Gamma','210','198','225','633'))
    for ($r = 0; $r -lt $rows.Count; $r++) {
        for ($c = 0; $c -lt 5; $c++) {
            Paint-Rect $g $(if (($r % 2) -eq 0) {'#FFFFFF'} else {'#F5F7FA'}) ($x + $c*$cellW) ($y + ($r+1)*$cellH) $cellW $cellH
            Paint-Text $g $rows[$r][$c] ($x + $c*$cellW + 18) ($y + ($r+1)*$cellH + 17) 20
        }
    }
    Paint-Text $g '最高合计：Gamma 633' 94 448 27 '#18794E' $true
}

Save-Fixture 'popup-error-dialog.png' {
    param($g)
    Paint-Window $g '示例同步工具'
    Paint-Text $g '文件同步' 105 146 34 '#233044' $true
    Paint-Text $g '等待任务完成…' 105 205 22 '#667085'
    Paint-Rect $g '#C8D0DA' 324 196 650 332
    Paint-Rect $g '#FFFFFF' 316 188 650 332
    Paint-Rect $g '#E9EDF3' 316 188 650 54
    Paint-Text $g '连接失败' 338 202 22 '#1F2937' $true
    Paint-Text $g '无法连接到示例服务器。' 358 285 25 '#1F2937'
    Paint-Text $g '错误代码：NET-1042' 358 339 22 '#B42318' $true
    Paint-Rect $g '#2563EB' 760 444 154 48
    Paint-Text $g '确定' 812 454 20 '#FFFFFF' $true
}

Save-Fixture 'multi-window-desktop.png' {
    param($g)
    Paint-Window $g '背景：示例资料' '#FFFFFF' 45 45 710 500
    Paint-Text $g '项目资料（背景窗口）' 85 132 27 '#374151' $true
    Paint-Rect $g '#CCD5E1' 480 175 735 460
    Paint-Rect $g '#FFFFFF' 470 165 735 460
    Paint-Rect $g '#E8EDF4' 470 165 735 52
    Paint-Text $g '前台：下载管理器' 492 178 22 '#182230' $true
    Paint-Text $g '模型资源包' 514 276 27 '#1F2937' $true
    Paint-Text $g '下载完成  100%' 514 330 24 '#18794E' $true
    Paint-Rect $g '#D1FAE5' 514 386 620 26
}

Save-Fixture 'small-chinese-text.png' {
    param($g)
    Paint-Window $g '系统设置 — 通知'
    Paint-Text $g '通知设置' 100 126 34 '#182230' $true
    Paint-Text $g '允许应用发送通知' 100 201 22
    Paint-Rect $g '#2563EB' 1040 198 78 34
    Paint-Text $g '专注模式下仅显示优先通知，横幅将在五秒后自动隐藏。' 100 258 15 '#536172'
    Paint-Line $g '#D8DEE8' 1 100 305 1170 305
    Paint-Text $g '桌面助手' 100 348 21 '#182230' $true
    Paint-Text $g '声音、横幅和通知中心' 100 386 14 '#667085'
    Paint-Text $g '测试提示：小字号内容应准确识别，无法确认时不要猜测。' 100 454 13 '#667085'
}

Save-Fixture 'dark-theme-ui.png' {
    param($g)
    Paint-Window $g '监控面板 — 暗色主题' '#111827'
    Paint-Text $g '服务状态' 105 125 32 '#F3F4F6' $true
    Paint-Rect $g '#1F2937' 105 192 310 156
    Paint-Text $g 'API 网关' 132 218 21 '#D1D5DB'
    Paint-Text $g '正常' 132 273 30 '#34D399' $true
    Paint-Rect $g '#1F2937' 448 192 310 156
    Paint-Text $g '任务队列' 475 218 21 '#D1D5DB'
    Paint-Text $g '积压 12' 475 273 30 '#FBBF24' $true
    Paint-Text $g '最近告警：无' 105 405 22 '#9CA3AF'
}

Save-Fixture 'scrolling.png' {
    param($g)
    Paint-Window $g '示例视频课程'
    Paint-Rect $g '#172033' 92 122 770 432
    Paint-Rect $g '#2D6A8A' 132 168 690 270
    Paint-Text $g '离线模型性能测试' 214 262 38 '#FFFFFF' $true
    Paint-Text $g '第 3 章：首 token 延迟' 245 326 25 '#D7ECF7'
    Paint-Text $g '▶  08:24 / 18:00' 118 501 19 '#FFFFFF'
    Paint-Text $g '课程目录' 902 125 25 '#1F2937' $true
    Paint-Text $g '1. 环境准备' 902 184 19
    Paint-Text $g '2. 模型加载' 902 232 19
    Paint-Rect $g '#E8F0FE' 884 272 286 52
    Paint-Text $g '3. 首 token 延迟' 902 286 19 '#2457C5' $true
    Paint-Text $g '4. 结果分析' 902 350 19
}
