inputDir = getDirectory("Choose Input Directory"); // 选择输入文件夹
outputDir = getDirectory("Choose Output Directory"); // 选择输出文件夹

fileList = getFileList(inputDir); // 获取文件列表

for (i = 0; i < fileList.length; i++) {
    if (endsWith(fileList[i], ".jpg")) { // 只处理JPG文件
        showProgress(i+1, fileList.length); // 显示进度
        
        // 打开原始图像
        open(inputDir + fileList[i]);
        originalTitle = getTitle();
        
        // 执行处理流程
        run("Split Channels");
        blueTitle = originalTitle + " (blue)";
        selectImage(blueTitle);
        run("Invert");
        run("Subtract Background...", "rolling=10 sliding");
        
        // 保存处理后的蓝色通道
        saveAs("Tiff", outputDir + "processed_" + fileList[i]);
        
        // 清理窗口
        close(originalTitle); // 关闭原始图像
        close(originalTitle + " (red)"); // 关闭红色通道
        close(originalTitle + " (green)"); // 关闭绿色通道
        close(blueTitle); // 关闭处理后的蓝色通道
    }
}
showStatus("Processing complete!"); // 完成提示