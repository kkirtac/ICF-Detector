#include "features.hpp"
#include "utils.hpp"
#include <algorithm>
#include <boost/algorithm/string.hpp>

using std::max;
using std::min;

using std::vector;
using std::string;

using std::make_pair;

using std::tuple;
using std::get;
using std::tie;
using std::make_tuple;

using std::mt19937;
using std::uniform_int_distribution;

using std::ifstream;
using std::ofstream;
using std::endl;

using cv::Mat;
using cv::Mat_;
using cv::Vec3b;
using cv::Vec6f;
using cv::Size;
using cv::Range;

using cv::imread;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


vector<Mat>
integrate_channels(const vector<Mat> &channels)
{
	vector<Mat> integral_channels;
	Mat ch;
	for (const Mat &channel : channels) {
		integral(channel, ch, CV_64F);
		integral_channels.push_back(ch.clone());
	}
	return integral_channels;
}

vector<Mat>
compute_channels(const Mat &image)
{

	vector<Mat> channels;

	//	try{
	//
	//		CV_Assert(image.rows > 0 && image.cols > 0);
	//	}
	//	catch(cv::Exception e)
	//	{
	//		std::cerr << "Exception while computing channel" << endl;
	//		std::cerr << "Error message: " << e.msg << endl;
	//
	//		return channels;
	//	}


	Mat src(image.rows, image.cols, CV_32FC3);
	image.convertTo(src, CV_32FC3, 1./255);

	Mat_<float> grad;
	Mat gray;
	cvtColor(src, gray, CV_RGB2GRAY);

	Mat_<float> row_der, col_der;
	Sobel(gray, row_der, CV_32F, 0, 1);
	Sobel(gray, col_der, CV_32F, 1, 0);

	magnitude(row_der, col_der, grad);

	Mat_<Vec6f> hist(grad.rows, grad.cols);

	for (int row = 0; row < grad.rows; ++row) {

		for (int col = 0; col < grad.cols; ++col) {

			for (int i = 0; i < 6; i++)
			{
				hist(row, col)[i] = 0.f;
			}

		}

	}


	const float to_deg = 180 / 3.1415926;
	for (int row = 0; row < grad.rows; ++row) {
		for (int col = 0; col < grad.cols; ++col) {
			float angle = atan2(row_der(row, col), col_der(row, col)) * to_deg;

			if (angle < 0)
				angle += 180;

			int ind = angle / 30;

			// indeks maks 5 olmali.
			// angle 180 olabiliyor,
			// bu durumda indeks 5'i geciyor.
			if(ind > 5)
			{
				ind = 5;
			}
			hist(row, col)[ind] = grad(row, col);
		}

	}


	channels.push_back(gray);
	channels.push_back(grad);

	vector<Mat> hist_channels;
	split(hist, hist_channels);

	for (const Mat &h : hist_channels)
		channels.push_back(h);


	return channels;
}

int rand0()
{
	int dummy = rand();
	while (dummy == 0)
	{
		dummy = rand();
	}

	return dummy;
}

typedef uniform_int_distribution<> uniform;

mt19937 gen(rand0());

tuple<int, int, int, int, int>
generate_feature(int channel_count, const tuple<int, int> &win_size,
		const tuple<int, int> &min_size)
		{

	uniform channel_gen(0, channel_count - 1);
	int channel = channel_gen(gen);

	uniform row_gen(0, get<0>(win_size) - get<0>(min_size) - 1);
	int row = row_gen(gen);

	uniform col_gen(0, get<1>(win_size) - get<1>(min_size) - 1);
	int col = col_gen(gen);

	uniform row_end_gen(row + get<0>(min_size), get<0>(win_size) - 1);
	int row_end = row_end_gen(gen);

	uniform col_end_gen(col + get<1>(min_size), get<1>(win_size) - 1);
	int col_end = col_end_gen(gen);

	return make_tuple(channel, row, col, row_end, col_end);
		}

vector<tuple<int, int, int, int, int>>
generate_features(int channel_count, const tuple<int, int> &win_size,
		const tuple<int, int> &min_size, int feature_count)
		{
	vector<tuple<int, int, int, int, int>> features;
	for (int i = 0; i < feature_count; ++i)
		features.push_back(generate_feature(channel_count, win_size, min_size));

	return features;
		}

double
extract_feature_sum(const vector<Mat> &channels,
		const tuple<int, int, int, int, int> &feature)
{
	int chan, row, col, row_end, col_end;
	tie(chan, row, col, row_end, col_end) = feature;

	assert(channels[chan].type() == CV_32F);
	Mat_<float> ch = channels[chan];

	double sum = 0;
	for (int r = row; r <= row_end; ++r)
		for (int c = col; c <= col_end; ++c)
			sum += ch(r, c);

	return sum;
}


double
extract_integral_feature(const vector<Mat> &integral_channels,
		const tuple<int, int, int, int, int> &feature,
		const tuple<int, int> &pin = tuple<int, int>(0, 0))
{
	int chan, row, col, row_end, col_end;
	tie(chan, row, col, row_end, col_end) = feature;
	int pin_row, pin_col;
	tie(pin_row, pin_col) = pin;
	row += pin_row;
	row_end += pin_row;
	col += pin_col;
	col_end += pin_col;

	const Mat_<double> ch = integral_channels[chan];
	auto res = ch(row_end + 1, col_end + 1) - ch(row, col_end + 1) -
			ch(row_end + 1, col) + ch(row, col);
	return res;
}

Mat_<float>
extract_integral_features(const vector<Mat> &integral_channels,
		const vector<tuple<int, int, int, int, int>> &features,
		const tuple<int, int> &pin = tuple<int, int>(0, 0))
		{
	Mat_<float> feature_vals(1, features.size());
	for (int i = 0; i < feature_vals.cols; ++i)
		feature_vals(0, i) = extract_integral_feature(integral_channels, features[i], pin);
	return feature_vals;
		}

Mat_<float>
compute_features(const Mat &image,
		const vector<tuple<int, int, int, int, int>> &features, int label, string labelfilename,
		string filename)
		{
	auto integral_channels = integrate_channels(compute_channels(image));
	Mat_<float> feature_vals(1, features.size());

	float val;

	ofstream labelout(labelfilename.c_str(), std::ios_base::app);
	labelout << label << endl;

	ofstream fout(filename.c_str(), std::ios_base::app);

	for (int i = 0; i < feature_vals.cols; ++i){
		val = extract_integral_feature(integral_channels, features[i]);
		fout << val << " " ;
		feature_vals(0, i) = val;
	}

	fout << endl ;

	return feature_vals;

}

tuple<Mat_<float>, Mat_<float>>
sample_train_features(const string &pos_dir_path, const string &neg_dir_path,
		const vector<tuple<int, int, int, int, int>> &features)
		{
	vector<string> extensions;
	extensions.push_back(".png");
	extensions.push_back(".jpg");
	extensions.push_back(".ppm");
	auto pos_filenames = list_files(pos_dir_path, extensions);
	auto neg_filenames = list_files(neg_dir_path, extensions);

	vector<string> filenames = pos_filenames;
	filenames.insert(filenames.end(),
			neg_filenames.begin(), neg_filenames.end());

	int n_pos = pos_filenames.size();
	int n_neg = neg_filenames.size();
	int n_samples = n_pos + n_neg;

	Mat_<float> labels(1, n_samples);
	labels.colRange(0, n_pos) = +1;
	labels.colRange(n_pos, n_samples) = -1;

	Mat_<float> feature_vals(n_samples, features.size());
#pragma omp parallel for
for (int i = 0; i < feature_vals.rows; ++i) {
	Mat image = imread(filenames[i]);
	compute_features(image(cv::Rect(16,16,64,128)), features, 1, "", "").copyTo(feature_vals.row(i));
}
return make_tuple(feature_vals, labels);
		}


tuple<cv::Mat_<float>, cv::Mat_<float>>
sample_train_features(const std::string &pos_dir_path, const std::string &neg_dir_path,
		int neg_win_per_img, const cv::Size neg_win_size,
		const std::vector<std::tuple<int, int, int, int, int>> &features,
		string filename, int numpos, int numneg)
		{
	vector<string> extensions;
	extensions.push_back(".png");
	extensions.push_back(".jpg");
	extensions.push_back(".ppm");
	auto pos_filenames = list_files(pos_dir_path, extensions);
	auto neg_filenames = list_files(neg_dir_path, extensions);

	if(numpos>0)
	{
		vector<int> pos_indices;
		for(int i=0; i<pos_filenames.size()-numpos; i++){
			uniform pos_idx_gen(0, pos_filenames.size()-1);
			pos_indices.push_back(pos_idx_gen(gen));
		}

		sort(pos_indices.begin(), pos_indices.end());

		for(int i=pos_indices.size()-1; i>=0; i--){
			pos_filenames.erase(pos_filenames.begin()+pos_indices[i]);
		}
	}

	if(numneg>0)
	{
		vector<int> neg_indices;
		for(int i=0; i<neg_filenames.size()-numneg; i++){
			uniform neg_idx_gen(0, neg_filenames.size()-1);
			neg_indices.push_back(neg_idx_gen(gen));
		}

		sort(neg_indices.begin(), neg_indices.end());

		for(int i=neg_indices.size()-1; i>=0; i--){
			neg_filenames.erase(neg_filenames.begin()+neg_indices[i]);
		}
	}

	vector<int> neg_indices;
	for(int i=0; i<neg_filenames.size(); i++){
		neg_indices.push_back(i);
	}
	std::random_shuffle( neg_indices.begin(), neg_indices.end() );


	vector<string> filenames = pos_filenames;
	filenames.insert(filenames.end(),
			neg_filenames.begin(), neg_filenames.end());

	int n_pos = pos_filenames.size();

	// negatifleri cogaltmak icin her negatif goruntuden
	// neg_win_per_img kadar random window sececegiz
	int n_neg = neg_filenames.size() * neg_win_per_img;
	int n_samples = n_pos + n_neg;

	std::cout << "number of positives: " << n_pos << endl;
	std::cout << "number of negatives: " << n_neg << endl;

	Mat_<float> labels(1, n_samples);
	labels.colRange(0, n_pos) = +1;
	labels.colRange(n_pos, n_samples) = -1;

	Mat_<float> feature_vals(n_samples, features.size());
	//#pragma omp parallel for
	Mat image, image_neg_crop, image_resize;

	int j=0;

	string filename_bootstrap;
	std::vector<std::string> strs;
	boost::split(strs, filename, boost::is_any_of("."));
	string bstrap = ".bootstrapfeaturevals";
	string lbls = ".labels";
	string lbls2 = ".bootstraplabels";
	string new_filename = "";
	string labels_filename = "";
	string labels_filename2 = "";
	for(int i=0; i<strs.size()-1; i++){
		new_filename += strs[i];
		labels_filename += strs[i];
		labels_filename2 += strs[i];
	}
	new_filename += bstrap;
	labels_filename += lbls;
	labels_filename2 += lbls2;

	int numbootstrap = 200;


	for (int i=0; i < pos_filenames.size(); ++i) { //filenames.size();

		image = imread(pos_filenames[i]); //filenames[i]

		std::cout << "processing pos sample " << i << " / " << pos_filenames.size()-1 << endl;


		compute_features(image(cv::Rect((image.size().width-neg_win_size.width)/2,
				(image.size().height-neg_win_size.height)/2,neg_win_size.width,neg_win_size.height)),
				features, 1, labels_filename, filename).copyTo(feature_vals.row(j));
		j++;
	}

	int c = 0;
	for(vector<int>::iterator it=neg_indices.begin(); it!=neg_indices.end(); ++it){

		image = imread(neg_filenames[*it]);

		std::cout << "processing neg sample " << c << " / " << neg_filenames.size()-1 << endl;

//		// negatif kumeden numbootstrap kadarini bootstrap sample olarak ayir
//		if(c < numbootstrap){
//
//			for(int k=0; k<50; k++)
//			{
//				uniform row_gen(0, image.size().height-neg_win_size.height);
//				int row_start = row_gen(gen);
//
//				uniform col_gen(0, image.size().width-neg_win_size.width);
//				int col_start = col_gen(gen);
//
//				uniform crop_height_gen(neg_win_size.height, image.size().height-row_start);
//				int crop_height = crop_height_gen(gen);
//
//				uniform crop_width_gen(neg_win_size.width, image.size().width-col_start);
//				int crop_width = crop_width_gen(gen);
//
//				image_neg_crop = image(cv::Rect(col_start, row_start, crop_width, crop_height));
//				resize(image_neg_crop, image_resize, neg_win_size);
//				compute_features(image_resize, features, -1, labels_filename2, new_filename);
//			}
//		}
//		// bootstrap olarak ayirdiklarimiz haricindeki negatifleri
//		// egitim kumesine ekle
//		else{

			// her neg image icin, neg_win_per_img kadar window
			// rasgele secilip crop edilecek
			for(int k=0; k<neg_win_per_img; k++)
			{
				uniform row_gen(0, image.size().height-neg_win_size.height);
				int row_start = row_gen(gen);

				uniform col_gen(0, image.size().width-neg_win_size.width);
				int col_start = col_gen(gen);

				uniform crop_height_gen(neg_win_size.height, image.size().height-row_start);
				int crop_height = crop_height_gen(gen);

				uniform crop_width_gen(neg_win_size.width, image.size().width-col_start);
				int crop_width = crop_width_gen(gen);

				image_neg_crop = image(cv::Rect(col_start, row_start, crop_width, crop_height));
				resize(image_neg_crop, image_resize, neg_win_size);
				compute_features(image_resize, features, -1, labels_filename, filename).copyTo(feature_vals.row(j));
				j++;
			}

		//}

		c++;

	}

	return make_tuple(feature_vals, labels);

}


void
save_features(const vector<tuple<int, int, int, int, int>> &features,
		const string &filename)
{
	ofstream fout(filename);
	int ch, r, c, re, ce;
	for (const auto &feature : features) {
		tie(ch, r, c, re, ce) = feature;
		fout << ch << " " << r << " " << c << " " << re << " " << ce << endl;
	}
}

vector<tuple<int, int, int, int, int>>
load_features(const string &filename)
{
	ifstream fin(filename);
	vector<tuple<int, int, int, int, int>> features;
	int ch, r, c, re, ce;
	while ((fin >> ch >> r >> c >> re >> ce))
		features.push_back(make_tuple(ch, r, c, re, ce));
	return features;
}

void test_feature_generation(const string &image_path)
{
	int channel_count = 8;
	int win_rows = 128;
	int win_cols = 64;
	int min_rows = 4;
	int min_cols = 4;
	int feature_count = 1000000;
	auto features = generate_features(channel_count,
			make_tuple(win_rows, win_cols), make_tuple(min_rows, min_cols),
			feature_count);

	int channel, row, col, row_end, col_end;
	for (const auto &feature : features) {
		tie(channel, row, col, row_end, col_end) = feature;
		assert(0 <= channel && channel < channel_count);

		assert(0 <= row && row < win_rows);
		assert(0 <= row_end && row_end < win_rows);
		assert(row < row_end && (row_end - row) >= min_rows);

		assert(0 <= col && col < win_cols);
		assert(0 <= col_end && col_end < win_cols);
		assert(col < col_end && (col_end - col) >= min_cols);
	}

	auto channels = compute_channels(imread(image_path));
	auto integral_channels = integrate_channels(channels);

	for (const auto &feature : features) {
		double integral_sum = extract_integral_feature(integral_channels, feature);
		double sum = extract_feature_sum(channels, feature);

		assert(abs(integral_sum - sum) < 1e-9);
	}
}
