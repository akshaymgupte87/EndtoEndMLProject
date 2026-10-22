param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PytestArgs = @()
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$drive = $repoRoot.Substring(0, 1).ToLowerInvariant()
if ($repoRoot -notmatch '^[A-Za-z]:\\') {
    throw "Expected a drive-mounted repository path, got: $repoRoot"
}
$repoRootInWsl = "/mnt/$drive" + ($repoRoot.Substring(2) -replace '\\', '/')

$linuxPython = '/home/akshay/miniconda3/bin/python'
$linuxJavaHome = '/usr/lib/jvm/java-21-openjdk-amd64'
$linuxSparkHome = '/home/akshay/miniconda3/lib/python3.11/site-packages/pyspark'

$wslArgs = @(
    '-d', 'Ubuntu', '--cd', $repoRootInWsl, '--exec', '/usr/bin/env',
    "JAVA_HOME=$linuxJavaHome",
    "SPARK_HOME=$linuxSparkHome",
    "PATH=$linuxSparkHome/bin:/home/akshay/miniconda3/bin:$linuxJavaHome/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYSPARK_PYTHON=$linuxPython",
    "PYTHONPATH=$repoRootInWsl",
    'SPARK_LOCAL_IP=127.0.0.1',
    'PYSPARK_SUBMIT_ARGS=--driver-memory 8g pyspark-shell',
    $linuxPython, '-m', 'pytest', '-o', 'cache_dir=/tmp/endtoendml-pytest-cache'
) + $PytestArgs

& wsl.exe @wslArgs
exit $LASTEXITCODE
