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

// ��ȫ�ͷ�ָ��
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
// �����˲���
PixelFilter(unsigned short* depthArray, unsigned short* smoothDepthArray, int innerBandThreshold = 3, int outerBandThreshold = 7)
{
	// ������������ֵ��ȷ����������ȷ�ķ�Χ��
	int widthBound = 512 - 1;
	int heightBound = 424 - 1;

	// ����ÿ������
	for (int depthArrayRowIndex = 0; depthArrayRowIndex<424; depthArrayRowIndex++)
	{
		// ����һ�������е�ÿ������
		for (int depthArrayColumnIndex = 0; depthArrayColumnIndex < 512; depthArrayColumnIndex++)
		{
			int depthIndex = depthArrayColumnIndex + (depthArrayRowIndex * 512);

			// ������Ϊ���ֵΪ0�����ؼ�Ϊ��ѡ����
			if (depthArray[depthIndex] == 0)
			{
				// ͨ���������������ǿ��Լ���õ����صĺ�������
				int x = depthIndex % 512;
				int y = (depthIndex - x) / 512;

				// filter collection ���������˲�����ÿ�����ֵ��Ӧ��Ƶ�ȣ��ں���
				// ���ǽ�ͨ�������ֵ��ȷ������ѡ����һ��ʲô���ֵ��
				unsigned short filterCollection[24][2] = { 0 };

				// �������ڷ��������������������ں�������ȷ����ѡ�����Ƿ��˲�
				int innerBandCount = 0;
				int outerBandCount = 0;

				// �����ѭ��������Ժ�ѡ����Ϊ���ĵ�5 X 5���������н��б��������ﶨ���������߽硣�����
				// ��������ڵ�����Ϊ���㣬��ô���ǽ���¼������ֵ�����������ڱ߽�ļ�������һ�����������
				// �߹��趨����ֵ����ô���ǽ�ȡ�˲�����ͳ�Ƶ����ֵ��������Ƶ����ߵ��Ǹ����ֵ��Ӧ���ں�ѡ
				// ������
				for (int yi = -2; yi < 3; yi++)
				{
					for (int xi = -2; xi < 3; xi++)
					{
						// yi��xiΪ������������ں�ѡ���ص�ƽ����

						// ���ǲ�Ҫxi = 0&&yi = 0���������Ϊ��ʱ�����ľ��Ǻ�ѡ����
						if (xi != 0 || yi != 0)
						{
							// ȷ���������������ͼ�е�λ��
							int xSearch = x + xi;
							int ySearch = y + yi;

							// ���������ص�λ���Ƿ񳬹���ͼ��ı߽磨��ѡ������ͼ��ı�Ե��
							if (xSearch >= 0 && xSearch <= widthBound &&
								ySearch >= 0 && ySearch <= heightBound)
							{
								int index = xSearch + (ySearch * 512);
								// ����ֻҪ������
								if (depthArray[index] != 0)
								{
									// ����ÿ�����ֵ��Ƶ��
									for (int i = 0; i < 24; i++)
									{
										if (filterCollection[i][0] == depthArray[index])
										{
											// ����� filter collection���Ѿ���¼����������
											// �������ȶ�Ӧ��Ƶ�ȼ�һ
											filterCollection[i][1]++;
											break;
										}
										else if (filterCollection[i][0] == 0)
										{
											// ���filter collection��û�м�¼������
											// ��ô��¼
											filterCollection[i][0] = depthArray[index];
											filterCollection[i][1]++;
											break;
										}
									}

									// ȷ���������ĸ��߽��ڵ����ز�Ϊ�㣬����Ӧ��������һ
									if (yi != 2 && yi != -2 && xi != 2 && xi != -2)
										innerBandCount++;
									else
										outerBandCount++;
								}
							}
						}
					}
				}

				// �жϼ������Ƿ񳬹���ֵ�����������ڷ������ص���Ŀ��������ֵ��
				// ��Ҫ�����з����������ֵ��Ӧ��ͳ������
				if (innerBandCount >= innerBandThreshold || outerBandCount >= outerBandThreshold)
				{
					short frequency = 0;
					short depth = 0;
					// ���ѭ����ͳ�����з����������ֵ��Ӧ������
					for (int i = 0; i < 24; i++)
					{
						// ��û�м�¼���ֵʱ���޷������ֵ�����أ�
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
				// ������ص����ֵ��Ϊ�㣬����ԭ���ֵ
				smoothDepthArray[depthIndex] = depthArray[depthIndex];
			}
		}
	}
}

Mat
// ��һ�����ͼ���ӻ�
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
#pragma region ��ʼ���豸
	// ��ȡKinect�豸
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
			// ��ȡ������Դ����ȡ��  
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
	// ��������֡������
	IDepthFrameReference* m_pDepthFrameReference = NULL;
	IColorFrameReference* m_pColorFrameReference = NULL;
	IDepthFrame* m_pDepthFrame = NULL;
	IColorFrame* m_pColorFrame = NULL;
	// �ĸ���ͼƬ��ʽ
	Mat i_rgb(1080, 1920, CV_8UC4);      //ע�⣺�������Ϊ4ͨ����ͼ��Kinect������ֻ����Bgra��ʽ����
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
	// �洢ǰN֡�������
	std::vector<UINT16*> queDepthArrays;
#pragma region ��ȡ����֡�߳�
	std::mutex mutex1;
	std::thread th1 = thread([&]{
		while (1)
		{
			mutex1.lock();
			vector<float> cloud;
			HRESULT hr = 0;
			// ��ȡ�µ�һ����Դ����֡
			hr = m_pMultiFrameReader->AcquireLatestFrame(&m_pMultiFrame);	// ���m_pMultiFrame��Ϊ�գ����Ὣ���ÿ�
			if (m_pMultiFrame == NULL)
			{
				mutex1.unlock();
				continue;
			}
			
			// �Ӷ�Դ����֡�з������ɫ���ݣ�������ݺͺ�������
			if (SUCCEEDED(hr))
				hr = m_pMultiFrame->get_ColorFrameReference(&m_pColorFrameReference);
			if (SUCCEEDED(hr))
				hr = m_pColorFrameReference->AcquireFrame(&m_pColorFrame);
			if (SUCCEEDED(hr))
				hr = m_pMultiFrame->get_DepthFrameReference(&m_pDepthFrameReference);
			if (SUCCEEDED(hr))
				hr = m_pDepthFrameReference->AcquireFrame(&m_pDepthFrame);


			// color������ͼƬ��
			UINT nColorBufferSize = 1920 * 1080 * 4;
			if (SUCCEEDED(hr))
				hr = m_pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, bgraData, ColorImageFormat::ColorImageFormat_Bgra);

			// ���depth������
			UINT nDepthBufferSize = 424 * 512;
			if (SUCCEEDED(hr))
			{
				hr = m_pDepthFrame->CopyFrameDataToArray(nDepthBufferSize, depthData); 
			}

			// �����˲���
			PixelFilter(depthData, pixelFilterData, 3, 5);

			// ��Ȩ�ƶ�ƽ��
			// ǰ֡����N
			int N = 5;
			// �����������ֵ�ܺ�
			UINT16 sumDepthData[424 * 512] = { 0 };
			// ��ʼ������ǰN֡�����
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
					// �����м��뵱ǰ֡
					UINT16 *temp = new UINT16[424 * 512];
					memcpy(temp, depthData, 424 * 512 * 2);
					queDepthArrays.push_back(temp);
				}
				else
				{
					// �����м��뵱ǰ֡��queDepthArrays.back()�൱�ڵ�ǰ֡�����ֻ��Ҫ������
					memcpy(queDepthArrays.back(), depthData, 424 * 512 * 2);
				}
				
				// �������Ѵ���N��ǰ֡+һ����ǰ֡
				// Denominator��ʾ��ĸ��count��ʾȨֵ�ķ���
				int Denominator = 0;
				int Count = 1;

				// �������ȴ���һ���յ����ͼ��ÿ������λ���ϴ���ǰN֡�ӵ�ǰ֡���ֵ��Ȩ֮��
				// ���ÿ�����س���Ȩֵ֮��
				for each (auto item in queDepthArrays)
				{
					// ����ÿ������
					for (int depthArrayRowIndex = 0; depthArrayRowIndex < 424; depthArrayRowIndex++)
					{
						// ����ÿ������
						for (int depthArrayColumnIndex = 0; depthArrayColumnIndex < 512; depthArrayColumnIndex++)
						{
							int index = depthArrayColumnIndex + (depthArrayRowIndex * 512);
							sumDepthData[index] += item[index] * Count;
						}
					}
					Denominator += Count;
					Count++;
				}

				// ����Ȩֵ֮��
				for (int i = 0; i<512 * 424;i++) 
				{
					queDepthArrays.back()[i] = depthData[i];
					averagedDepthData[i] = (short)(sumDepthData[i] / Denominator);
				}
				// �൱��queue.pop()
				auto temp = queDepthArrays.begin();
				for (auto iter = queDepthArrays.begin(); iter !=queDepthArrays.end()-1; iter++)
				{
					*iter = *(iter + 1);
				}
				queDepthArrays.back() = *temp;
			}
			i_average.data = (unsigned char *)averagedDepthData;
#pragma region ��ʾͼƬ
			if (SUCCEEDED(hr))
			{
				//i_rgb.data = bgraData;
				//// ��ʾͼƬ
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

#pragma region ����ͼƬ
#ifdef SAVE_IMG
			imwrite("depth.png", i_depth_raw);
			imwrite("color.jpg", i_rgb);
			imwrite("depth2rgb.jpg", i_depthToRgb);
#endif
#pragma endregion
			// �ͷ���Դ
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
	// �رմ��ڣ��豸
	cv::destroyAllWindows();
	delete[] depthData;
	delete[] bgraData;
	delete[] pixelFilterData;

	SafeRelease(m_pCoordinateMapper);
	m_pKinectSensor->Close();
	std::system("pause");
	return 0;
}
