@echo off
IF "%ANACONDA_ROOT%"=="" (start D:\Programs\Anaconda\Scripts\activate.bat 2dasl) ELSE (start %ANACONDA_ROOT%\Scripts\activate.bat 2dasl)