"""
项目技术分析

google-drive-downloader的核心在于其简洁的API设计。

通过导入GoogleDriveDownloader类，你可以轻松地获取并下载Google Drive上的文件。

它主要依赖于以下功能：

download_file_from_google_drive()：

这是库的主要方法，接收三个参数：

file_id（Google Drive文件的唯一标识符），dest_path（目标保存路径）和unzip（是否解压缩）。

通过这个方法，你可以直接从共享链接下载文件，并选择是否自动解压。

showsize 和 overwrite 参数：这两个选项提供了额外的功能，如显示下载进度和覆盖已存在的同名文件。
"""

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(
    file_id="1sL9VjWAjPWAS2ilACqcR5WHLCjcF1u4k",
    dest_path="dataset/ena.pth",
    unzip=True,
)
