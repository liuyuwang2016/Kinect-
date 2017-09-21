#include <time.h>
#include "kinect.h"
#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <thread>
#include <mutex>
//#define GL_DISPLAY
//#define SAVE_IMG
using namespace cv;
using namespace std;

// 安全释放指针
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}

void
// 像素滤波器
PixelFilter(unsigned short* depthArray, unsigned short* smoothDepthArray, int innerBandThreshold = 3, int outerBandThreshold = 7)
{
	// 我们用这两个值来确定索引在正确的范围内
	int widthBound = 512 - 1;
	int heightBound = 424 - 1;

	// 处理每行像素
	for (int depthArrayRowIndex = 0; depthArrayRowIndex<424; depthArrayRowIndex++)
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
				unsigned short filterCollection[24][2] = { 0 };

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
}

Mat
// 将一个深度图可视化
ShowDepthImage(unsigned short* depthData)
{
	Mat result(424, 512, CV_8UC4);
	for (int i = 0; i < 512 * 424; i++)
	{
		UINT16 depthValue = depthData[i];
		if (depthValue == 0)
		{
			result.data[i * 4] = 255;
			result.data[i * 4 + 1] = 0;
			result.data[i * 4 + 2] = 0;
			result.data[i * 4 + 3] = depthValue % 256;
		}
		else
		{
			result.data[i * 4] = depthValue % 256;
			result.data[i * 4 + 1] = depthValue % 256;
			result.data[i * 4 + 2] = depthValue % 256;
			result.data[i * 4 + 3] = depthValue % 256;
		}
	}
	return result;
}

int main()
{
#pragma region 初始化设备
	// 获取Kinect设备
	IKinectSensor* m_pKinectSensor;
	ICoordinateMapper*      m_pCoordinateMapper;
	CameraIntrinsics* m_pCameraIntrinsics = new CameraIntrinsics();
	HRESULT hr;
	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	IMultiSourceFrameReader* m_pMultiFrameReader;
	IBodyFrameSource* m_pBodyFrameSource;
	IBodyFrameReader* m_pBodyFrameReader;
	if (m_pKinectSensor)
	{
		hr = m_pKinectSensor->Open();
		//Sleep(2000);
		if (SUCCEEDED(hr))
		{
			m_pKinectSensor->get_BodyFrameSource(&m_pBodyFrameSource);
			// 获取多数据源到读取器  
			hr = m_pKinectSensor->OpenMultiSourceFrameReader(
				FrameSourceTypes::FrameSourceTypes_Color |
				FrameSourceTypes::FrameSourceTypes_Infrared |
				FrameSourceTypes::FrameSourceTypes_Depth,
				&m_pMultiFrameReader);
		}
	}
	if (SUCCEEDED(hr))
	{
		hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
	}
	if (!m_pKinectSensor || FAILED(hr))
	{
		return E_FAIL;
	}
	// 三个数据帧及引用
	IDepthFrameReference* m_pDepthFrameReference = NULL;
	IColorFrameReference* m_pColorFrameReference = NULL;
	IDepthFrame* m_pDepthFrame = NULL;
	IColorFrame* m_pColorFrame = NULL;
	// 四个个图片格式
	Mat i_rgb(1080, 1920, CV_8UC4);      //注意：这里必须为4通道的图，Kinect的数据只能以Bgra格式传出
	//Mat i_depth_raw(424, 512, CV_16UC1);


	IMultiSourceFrame* m_pMultiFrame = NULL;

	DepthSpacePoint*        m_pDepthCoordinates = NULL;
	ColorSpacePoint*        m_pColorCoordinates = NULL;
	CameraSpacePoint*        m_pCameraCoordinates = NULL;

#pragma endregion
	m_pColorCoordinates = new ColorSpacePoint[512 * 424];
	m_pCameraCoordinates = new CameraSpacePoint[512 * 424];
	UINT16 *pixelFilterData = new UINT16[424 * 512];
	UINT16 *averagedDepthData = new UINT16[424 * 512];
	
	BYTE *bgraData = new BYTE[1080 * 1920 * 4];
	UINT16 *depthData = new UINT16[424 * 512];
	new UINT16[424 * 512];
	Mat i_depth(424, 512, CV_16UC1);
	Mat i_average(424, 512, CV_16UC1);
	Mat i_before(424, 512, CV_8UC4);
	Mat i_pixFilter(424, 512, CV_8UC4);
	// 存储前N帧深度数据
	std::vector<UINT16*> queDepthArrays;
#pragma region 获取点云帧线程
	std::mutex mutex1;
	std::thread th1 = thread([&]{
		while (1)
		{
			mutex1.lock();
			vector<float> cloud;
			HRESULT hr = 0;
			// 获取新的一个多源数据帧
			hr = m_pMultiFrameReader->AcquireLatestFrame(&m_pMultiFrame);	// 如果m_pMultiFrame不为空，这句会将其置空
			if (m_pMultiFrame == NULL)
			{
				mutex1.unlock();
				continue;
			}
			
			// 从多源数据帧中分离出彩色数据，深度数据和红外数据
			if (SUCCEEDED(hr))
				hr = m_pMultiFrame->get_ColorFrameReference(&m_pColorFrameReference);
			if (SUCCEEDED(hr))
				hr = m_pColorFrameReference->AcquireFrame(&m_pColorFrame);
			if (SUCCEEDED(hr))
				hr = m_pMultiFrame->get_DepthFrameReference(&m_pDepthFrameReference);
			if (SUCCEEDED(hr))
				hr = m_pDepthFrameReference->AcquireFrame(&m_pDepthFrame);


			// color拷贝到图片中
			UINT nColorBufferSize = 1920 * 1080 * 4;
			if (SUCCEEDED(hr))
				hr = m_pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, bgraData, ColorImageFormat::ColorImageFormat_Bgra);

			// 获得depth的数据
			UINT nDepthBufferSize = 424 * 512;
			if (SUCCEEDED(hr))
			{
				hr = m_pDepthFrame->CopyFrameDataToArray(nDepthBufferSize, depthData); 
			}

			// 像素滤波器
			PixelFilter(depthData, pixelFilterData, 3, 5);

			// 加权移动平均
			// 前帧数量N
			int N = 5;
			// 各个像素深度值总和
			UINT16 sumDepthData[424 * 512] = { 0 };
			// 初始化，将前N帧入队列
			if (queDepthArrays.size() < N)
			{
				UINT16 *temp = new UINT16[424 * 512];
				memcpy(temp, depthData, 424 * 512 * 2);
				queDepthArrays.push_back(temp);
			}
			else
			{
				if (queDepthArrays.size() == N)
				{
					// 队列中加入当前帧
					UINT16 *temp = new UINT16[424 * 512];
					memcpy(temp, depthData, 424 * 512 * 2);
					queDepthArrays.push_back(temp);
				}
				else
				{
					// 队列中加入当前帧，queDepthArrays.back()相当于当前帧，因此只需要更新它
					memcpy(queDepthArrays.back(), depthData, 424 * 512 * 2);
				}
				
				// 队列中已存满N个前帧+一个当前帧
				// Denominator表示分母，count表示权值的分子
				int Denominator = 0;
				int Count = 1;

				// 我们首先创建一个空的深度图，每个像素位置上储存前N帧加当前帧深度值加权之和
				// 最后每个像素除以权值之和
				for each (auto item in queDepthArrays)
				{
					// 处理每行像素
					for (int depthArrayRowIndex = 0; depthArrayRowIndex < 424; depthArrayRowIndex++)
					{
						// 处理每个像素
						for (int depthArrayColumnIndex = 0; depthArrayColumnIndex < 512; depthArrayColumnIndex++)
						{
							int index = depthArrayColumnIndex + (depthArrayRowIndex * 512);
							sumDepthData[index] += item[index] * Count;
						}
					}
					Denominator += Count;
					Count++;
				}

				// 除以权值之和
				for (int i = 0; i<512 * 424;i++) 
				{
					queDepthArrays.back()[i] = depthData[i];
					averagedDepthData[i] = (short)(sumDepthData[i] / Denominator);
				}
				// 相当于queue.pop()
				auto temp = queDepthArrays.begin();
				for (auto iter = queDepthArrays.begin(); iter !=queDepthArrays.end()-1; iter++)
				{
					*iter = *(iter + 1);
				}
				queDepthArrays.back() = *temp;
			}
			i_average.data = (unsigned char *)averagedDepthData;
#pragma region 显示图片
			if (SUCCEEDED(hr))
			{
				//i_rgb.data = bgraData;
				//// 显示图片
				//imshow("rgb", i_rgb);
				//if (waitKey(1) == VK_ESCAPE)
				//break;
				i_depth.data = (BYTE*)depthData;
				imshow("depth", i_depth);
				if (waitKey(1) == VK_ESCAPE)
					break;
				imshow("before", ShowDepthImage(depthData));
				if (waitKey(1) == VK_ESCAPE)
					break;
				imshow("pixFilter", ShowDepthImage(pixelFilterData));
				if (waitKey(1) == VK_ESCAPE)
					break;
				imshow("average", ShowDepthImage(averagedDepthData));
				if (waitKey(1) == VK_ESCAPE)
					break;
				/*imshow("depth2rgb", i_depthToRgb);
				if (waitKey(1) == VK_ESCAPE)
				break;*/
				/*imshow("depth_raw", i_depth_raw);
				if (waitKey(1) == VK_ESCAPE)
				break;*/
			}
#pragma endregion

#pragma region 保存图片
#ifdef SAVE_IMG
			imwrite("depth.png", i_depth_raw);
			imwrite("color.jpg", i_rgb);
			imwrite("depth2rgb.jpg", i_depthToRgb);
#endif
#pragma endregion
			// 释放资源
			SafeRelease(m_pColorFrame);
			SafeRelease(m_pDepthFrame);
			SafeRelease(m_pColorFrameReference);
			SafeRelease(m_pDepthFrameReference);
			SafeRelease(m_pMultiFrame);
			mutex1.unlock();
		}
	});
#pragma endregion
	th1.join();
	// 关闭窗口，设备
	cv::destroyAllWindows();
	delete[] depthData;
	delete[] bgraData;
	delete[] pixelFilterData;

	SafeRelease(m_pCoordinateMapper);
	m_pKinectSensor->Close();
	std::system("pause");
	return 0;
}
