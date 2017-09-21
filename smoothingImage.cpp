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
	// 1. ���ͼ����ʾ
	Mat i_depth(424, 512, CV_16UC1);
	i_depth = imread("depth.png", IMREAD_ANYDEPTH);	
	Mat i_before(424, 512, CV_8UC4);				// Ϊ����ʾ����
	Mat i_after(424, 512, CV_8UC4);				// Ϊ����ʾ����
	Mat i_result(424, 512, CV_16UC1);				// �˲����
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

	// 2. �����˲�
	
	// �˲������ͼ�Ļ���
	unsigned short* smoothDepthArray = (unsigned short*)i_result.data;
	// ������������ֵ��ȷ����������ȷ�ķ�Χ��
	int widthBound = 512 - 1;
	int heightBound = 424 - 1;

	// �ڣ�8�����أ��⣨16�����أ�����ֵ
	int innerBandThreshold = 3;
	int outerBandThreshold = 7;

	// ����ÿ������
	for (int depthArrayRowIndex = 0; depthArrayRowIndex<424;depthArrayRowIndex++)
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
				unsigned short filterCollection[24][2] = {0};

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

	// 3. ��ʾ
	thread th = std::thread([&]{
		while (true)
		{
			imshow("ԭʼ���ͼ", i_depth);
			waitKey(1);
			imshow("���ڹ۲��ԭʼ���ͼ", i_before);
			waitKey(1);
		}
	});

	thread th2 = std::thread([&]{
		while (true)
		{
			imshow("���ͼ", i_result);
			waitKey(1);
			imshow("���ڹ۲���˲����ͼ", i_after);
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