#include <iostream>
using std::cout;
using std::endl;

#include <string>
using std::string;
using std::stoi;

#include <vector>
using std::vector;

#include <utility>
using std::tuple;
using std::make_pair;

#include <tuple>
using std::get;
using std::tie;
using std::make_tuple;

#include <fstream>
using std::ifstream;
using std::ofstream;


#include <opencv2/core/core.hpp>
using cv::Mat;
using cv::Mat_;
using cv::Vec3b;
using cv::Size;
using cv::Range;

#include <ctime>

#include <opencv2/highgui/highgui.hpp>
using cv::imread;

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "features.hpp"
#include "detection.hpp"


Mat
visualize(const Mat &image, const vector<tuple<int, int, int, int>> &wins)
{
    CV_Assert(image.type() == CV_8UC3);
    Mat_<Vec3b> img = image.clone();
    for (const auto &win : wins) {
        int row, col, row_end, col_end;
        tie(row, col, row_end, col_end) = win;
        for (int i = row; i <= row_end; ++i) {
            img(i, col) = Vec3b(255, 0, 0);
            img(i, col_end) = Vec3b(255, 0, 0);
        }
        for (int i = col; i <= col_end; ++i) {
            img(row, i) = Vec3b(255, 0, 0);
            img(row_end, i) = Vec3b(255, 0, 0);
        }
    }
    return img;
}


int main(int argc, const char *argv[])
{
    //string pos_dir_path = "/tmp/inria/train/posWin/";
    //string neg_dir_path = "/tmp/inria/train/negWin/";

    //string test_img_path = "/tmp/inria/test/pos/0001.png";


//	//train
//	string pos_dir_path = "D:/INRIAPerson/train_64x128_H96/pos";
//    string neg_dir_path = "D:/INRIAPerson/Train/neg";

//	//bootstrap
//	string pos_dir_path = "D:/INRIAPerson/train_64x128_H96/pos";
//    string neg_dir_path = "D:/INRIAPerson/Bootstrap";

//	//calibration
//	string pos_dir_path = "D:/INRIAPerson/Calibration/pos";
//    string neg_dir_path = "D:/INRIAPerson/Calibration/neg";

	//test
	string pos_dir_path = "D:/INRIAPerson/test_64x128_H96/pos";
	string neg_dir_path = "D:/INRIAPerson/Test/neg";



    //string test_img_path = "D:/Datasets/INRIAPerson/test_64x128_H96/pos/crop001546b.png";

	string test_img_path;

    if (argc != 5 && argc != 6) {
        cout << "Usage: " << endl;
        cout << argv[0] << " feature_count weak_count max_depth step" << endl;
        cout << argv[0] << " features_filename model_filename step testImgPath outImgPath" << endl;
        return 0;
    }

    string filename;
    ofstream log;
    CvBoost boosting;
    double t;
    clock_t begin, end;

    int win_rows = 128; //96; //128
    int win_cols = 64; //38; //64
	int margin = 16;

    int step;

    vector<tuple<int, int, int, int, int>> features;
    if (argc == 5) {
        step = stoi(argv[4]);
        filename = string(argv[1]) + "_" + string(argv[2]) + "_" +
            string(argv[3]) + "_" + string(argv[4]);
        log.open(filename + ".txt");

        int feature_count = stoi(argv[1]);
        int weak_count = stoi(argv[2]);
        int max_depth = stoi(argv[3]);
        int channel_count = 8;
        int min_rows = 4;
        int min_cols = 4;

		cout << "Feature generation started.." << endl;

        features = generate_features(channel_count,
            make_tuple(win_rows, win_cols), make_tuple(min_rows, min_cols),
            feature_count);

        save_features(features, (filename + ".features").c_str());

        Mat_<float> feature_vals, labels;

        begin = clock();

//		int neg_win_per_img = 5;
//		int numpos=1000;
//		int numneg=500;
		int neg_win_per_img = 10;
		int numpos=0;
		int numneg=0;
		Size neg_win_size(win_cols, win_rows);

        tie(feature_vals, labels) = sample_train_features(pos_dir_path, neg_dir_path,
			neg_win_per_img, neg_win_size, features, filename + "_test" + ".featurevals", numpos, numneg);

		cout << "Feature generation finished.." << endl;

        end = clock();
        t = double(end - begin) / CLOCKS_PER_SEC;

        log << "Sample size: " << feature_vals.rows << endl;
        log << "Feature extraction: " << t << " sec" << endl;

        CvBoostParams params;
        params.max_depth = max_depth;
        params.weak_count = weak_count;

		cout << "Training started.." << endl;

        begin = clock();

        boosting.train(feature_vals, CV_ROW_SAMPLE, labels,
            Mat(), Mat(), Mat(), Mat(), params);

		cout << "Training finished.." << endl;

        end = clock();
        t = double(end - begin) / CLOCKS_PER_SEC;

        boosting.save((filename + ".model").c_str());
        log << "Boosting training: " << t << " sec" << endl;
    } else {
        step = stoi(argv[3]);
        //filename = argv[2];
        features = load_features(argv[1]);
        log.open(argv[2] + string(".txt"));
        boosting.load(argv[2]);
        log << "Model loaded" << endl;
		test_img_path = argv[4];
		filename = argv[5];
    }

	cout << "Detection started.." << endl;

    begin = clock();

    Mat test_image = imread(test_img_path);

	if (!test_image.data)
	{
		return -1;
	}


    vector<tuple<int, int, int, int, float>> wins;
    for (int scale_power = -10; scale_power <= 10; ++scale_power) {
        float scale = powf(2, scale_power / 10.);
        Mat image;
        resize(test_image, image, Size(), scale, scale);


		//for (int i = 0; i < features.size(); i++)
		//{
		//	int chan, row, col, row_end, col_end;
		//	tie(chan, row, col, row_end, col_end) = features[i];
		//	row *= scale;
		//	col *= scale;
		//	row_end *= scale;
		//	col_end *= scale;

		//	features.erase(features.begin()+i);

		//	features.insert(features.begin()+i, make_tuple(chan, row, col, row_end, col_end));
		//}

		//
		//int w_rows = win_rows * scale;
		//int w_cols = win_cols * scale;


        vector<Mat> integral_channels;
		for (int row = 0; row < image.rows - win_rows; row += step)
		{
			for (int col = 0; col < image.cols - win_cols; col += step)
			{
				integral_channels = integrate_channels(compute_channels(image));
                auto vals = extract_integral_features(integral_channels,
                    features, make_tuple(row, col));
                auto pred = boosting.predict(vals, Mat(), Range::all(), false, true);
                if (pred > 0)
				{
					int row_end = row + win_rows - 1;
					int col_end = col + win_cols - 1;
                    wins.push_back(make_tuple(row / scale, col / scale,
                        row_end / scale, col_end / scale, pred));
                }
            }
		}
    }

	cout << "Detection finished.." << endl;

    end = clock();
    t = double(end - begin) / CLOCKS_PER_SEC;
    log << "Image detection: " << t << " sec" << endl;


	auto img = visualize(test_image, nms(wins));


    imwrite(filename + ".png", img);

	return 0;
}
