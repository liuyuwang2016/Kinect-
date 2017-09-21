#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include "kinect.h"
using namespace cv;
using namespace std;

int main()
{
	// 1. 深度图并显示
	Mat i_depth(424, 512, CV_16UC1);
	i_depth = imread("depth.png", IMREAD_ANYDEPTH);	
	Mat i_before(424, 512, CV_8UC4);				// 为了显示方便
	Mat i_after(424, 512, CV_8UC4);				// 为了显示方便
	Mat i_result(424, 512, CV_16UC1);				// 滤波结果
	unsigned short maxDepth = 0; 
	unsigned short* depthArray = (unsigned short*)i_depth.data;
	unsigned short iZeroCountBefore = 0;
	unsigned short iZeroCountAfter = 0;
	for (int i = 0; i < 512 * 424; i++)
	{
		int row = i / 512;
		int col = i % 512;
		
		unsigned short depthValue = depthArray[row * 512 + col];
		if (depthValue == 0)
		{
			i_before.data[i * 4] = 255;
			i_before.data[i * 4 + 1] = 0;
			i_before.data[i * 4 + 2] = 0;
			i_before.data[i * 4 + 3] = depthValue / 256;
			iZeroCountBefore++;
		}
		else
		{
			i_before.data[i * 4] = depthValue / 8000.0f*256;
			i_before.data[i * 4 + 1] = depthValue / 8000.0f * 256;
			i_before.data[i * 4 + 2] = depthValue / 8000.0f* 256;
			i_before.data[i * 4 + 3] = depthValue / 8000.0f * 256;
		}
		maxDepth = depthValue > maxDepth ? depthValue : maxDepth;
	}
	cout << "max depth value: " << maxDepth << endl;

	// 2. 像素滤波
	
	// 滤波后深度图的缓存
	unsigned short* smoothDepthArray = (unsigned short*)i_result.data;
	// 我们用这两个值来确定索引在正确的范围内
	int widthBound = 512 - 1;
	int heightBound = 424 - 1;

	// 内（8个像素）外（16个像素）层阈值
	int innerBandThreshold = 3;
	int outerBandThreshold = 7;

	// 处理每行像素
	for (int depthArrayRowIndex = 0; depthArrayRowIndex<424;depthArrayRowIndex++)
	{
		// 处理一行像素中的每个像素
		for (int depthArrayColumnIndex = 0; depthArrayColumnIndex < 512; depthArrayColumnIndex++)
		{
			int depthIndex = depthArrayColumnIndex + (depthArrayRowIndex * 512);

			// 我们认为深度值为0的像素即为候选像素
			if (depthArray[depthIndex] == 0)
			{
				// 通过像素索引，我们可以计算得到像素的横纵坐标
				int x = depthIndex % 512;
				int y = (depthIndex - x) / 512;

				// filter collection 用来计算滤波器内每个深度值对应的频度，在后面
				// 我们将通过这个数值来确定给候选像素一个什么深度值。
				unsigned short filterCollection[24][2] = {0};

				// 内外层框内非零像素数量计数器，在后面用来确定候选像素是否滤波
				int innerBandCount = 0;
				int outerBandCount = 0;

				// 下面的循环将会对以候选像素为中心的5 X 5的像素阵列进行遍历。这里定义了两个边界。如果在
				// 这个阵列内的像素为非零，那么我们将记录这个深度值，并将其所在边界的计数器加一，如果计数器
				// 高过设定的阈值，那么我们将取滤波器内统计的深度值的众数（频度最高的那个深度值）应用于候选
				// 像素上
				for (int yi = -2; yi < 3; yi++)
				{
					for (int xi = -2; xi < 3; xi++)
					{
						// yi和xi为操作像素相对于候选像素的平移量

						// 我们不要xi = 0&&yi = 0的情况，因为此时操作的就是候选像素
						if (xi != 0 || yi != 0)
						{
							// 确定操作像素在深度图中的位置
							int xSearch = x + xi;
							int ySearch = y + yi;

							// 检查操作像素的位置是否超过了图像的边界（候选像素在图像的边缘）
							if (xSearch >= 0 && xSearch <= widthBound &&
								ySearch >= 0 && ySearch <= heightBound)
							{
								int index = xSearch + (ySearch * 512);
								// 我们只要非零量
								if (depthArray[index] != 0)
								{
									// 计算每个深度值的频度
									for (int i = 0; i < 24; i++)
									{
										if (filterCollection[i][0] == depthArray[index])
										{
											// 如果在 filter collection中已经记录过了这个深度
											// 将这个深度对应的频度加一
											filterCollection[i][1]++;
											break;
										}
										else if (filterCollection[i][0] == 0)
										{
											// 如果filter collection中没有记录这个深度
											// 那么记录
											filterCollection[i][0] = depthArray[index];
											filterCollection[i][1]++;
											break;
										}
									}

									// 确定是内外哪个边界内的像素不为零，对相应计数器加一
									if (yi != 2 && yi != -2 && xi != 2 && xi != -2)
										innerBandCount++;
									else
										outerBandCount++;
								}
							}
						}
					}
				}

				// 判断计数器是否超过阈值，如果任意层内非零像素的数目超过了阈值，
				// 就要将所有非零像素深度值对应的统计众数
				if (innerBandCount >= innerBandThreshold || outerBandCount >= outerBandThreshold)
				{
					short frequency = 0;
					short depth = 0;
					// 这个循环将统计所有非零像素深度值对应的众数
					for (int i = 0; i < 24; i++)
					{
						// 当没有记录深度值时（无非零深度值的像素）
						if (filterCollection[i][0] == 0)
							break;
						if (filterCollection[i][1] > frequency)
						{
							depth = filterCollection[i][0];
							frequency = filterCollection[i][1];
						}
					}

					smoothDepthArray[depthIndex] = depth;
				}
				else
				{
					smoothDepthArray[depthIndex] = 0;
				}
			}
			else
			{
				// 如果像素的深度值不为零，保持原深度值
				smoothDepthArray[depthIndex] = depthArray[depthIndex];
			}
		}
	}

	for (int i = 0; i < 512 * 424; i++)
	{
		int row = i / 512;
		int col = i % 512;

		unsigned short depthValue = smoothDepthArray[row * 512 + col];
		if (depthValue == 0)
		{
			i_after.data[i * 4] = 255;
			i_after.data[i * 4 + 1] = 0;
			i_after.data[i * 4 + 2] = 0;
			i_after.data[i * 4 + 3] = depthValue / 256;
			iZeroCountAfter++;
		}
		else
		{
			i_after.data[i * 4] = depthValue / 8000.0f * 256;
			i_after.data[i * 4 + 1] = depthValue / 8000.0f * 256;
			i_after.data[i * 4 + 2] = depthValue / 8000.0f * 256;
			i_after.data[i * 4 + 3] = depthValue / 8000.0f * 256;
		}
	}

	// 3. 显示
	thread th = std::thread([&]{
		while (true)
		{
			imshow("原始深度图", i_depth);
			waitKey(1);
			imshow("便于观察的原始深度图", i_before);
			waitKey(1);
		}
	});

	thread th2 = std::thread([&]{
		while (true)
		{
			imshow("结果图", i_result);
			waitKey(1);
			imshow("便于观察的滤波深度图", i_after);
			waitKey(1);
		}
	});
	cout << "iZeroCountBefore:    " << iZeroCountBefore << "  depthArray[0]:  "<<depthArray[0]<<endl;
	cout << "iZeroCountAfter:    " << iZeroCountAfter << "  smoothDepthArray[0]:  " << smoothDepthArray[0] <<endl;
	th.join();
	th2.join();
	std::system("pause");
	return 0;
}