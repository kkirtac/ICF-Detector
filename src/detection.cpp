#include "detection.hpp"

using std::max;
using std::min;

using std::vector;

using std::tuple;
using std::make_tuple;
using std::tie;
using std::get;

bool win_compare(const tuple<int, int, int, int, float> &win1,
                 const tuple<int, int, int, int, float> &win2)
{
    return get<4>(win1) < get<4>(win2);
}

vector<tuple<int, int, int, int>>
nms(const vector<tuple<int, int, int, int, float>> &wins_in)
{
    vector<tuple<int, int, int, int, float>> wins = wins_in;
    vector<tuple<int, int, int, int>> res;
    sort(wins.begin(), wins.end(), win_compare);
    int prob_r, prob_c, prob_re, prob_ce;
    int cur_r, cur_c, cur_re, cur_ce;
    float prob_max, cur_prob;
    while (!wins.empty()) {
        tie(prob_r, prob_c, prob_re, prob_ce, prob_max) = wins[wins.size() - 1];
        res.push_back(make_tuple(prob_r, prob_c, prob_re, prob_ce));

        int prob_area = (prob_re - prob_r + 1) * (prob_ce - prob_c + 1);
        vector<bool> suppress_mask(wins.size());
        suppress_mask[wins.size() - 1] = true;

        for (size_t i = 0; i < wins.size() - 1; ++i) {
            tie(cur_r, cur_c, cur_re, cur_ce, cur_prob) = wins[i];
            int cur_area = (cur_re - cur_r + 1) * (cur_ce - cur_c + 1);

            int overlap_r = max(cur_r, prob_r);
            int overlap_c = max(cur_c, prob_c);
            int overlap_re = min(cur_re, prob_re);
            int overlap_ce = min(cur_ce, prob_ce);
            int overlap_area = (overlap_re - overlap_r + 1)
                                * (overlap_ce - overlap_c + 1);

            if (overlap_area > 0 &&
                overlap_area / float(min(prob_area, cur_area)) > 0.5)
                    suppress_mask[i] = true;
        }
        vector<tuple<int, int, int, int, float>> wins_new;
        for (size_t i = 0; i < wins.size(); ++i) {
            if (!suppress_mask[i])
                wins_new.push_back(wins[i]);
        }
        wins = wins_new;
    }
    return res;
}
