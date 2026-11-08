**python版本为手敲，使用GPT将python转cpp**  
***
# 运行说明  
python main_video.py --image_path="图片路径"

# 函数作用说明(针对./rm_test_python/main_video.py,处理流程按照以下序号顺序)：
 **Dispose_image** 
1. **Get_mask**   
    1. **Blur_img**  图像去噪，使用高斯去噪，核大小为3*3  
    2. **Creat_mask_by_inRange**  使用inRange对图像进行二值化处  
        橙色HSV范围:[10, 76, 204]~[40, 255, 255]  
        为了防止图像边缘的红色部分缺失导致灯条过细，将橙色二值图并上红色二值图
    3. **Close_img** 形态学操作，使用闭运算，填补图像上方空洞  
2. **find_light**   
   1. 找灯条
   2. 检查灯条在**check_light**
3. **Match_light**  
    1. 遍历所有灯条
    2. 检查灯条部分在**check_light**   get_angle将旋转矩形角度统一，方便判断平行  
    3. 当**check_light**返回True时，为当前两灯条画框
    4. **sort_points**将每个灯条的随机顶点顺序统一为左上，左下，右上，右下，方便画出两个灯条的矩形框
    5. drawContours在result上画框  

**Dispose_video**  
视频处理部分  
输出视频在Output_video.avi  

### add:当处理不同颜色时，改变Creat_mask_by_inRange传入的color值，对原图像进行两次Dispose_image处理

