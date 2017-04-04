#include "utils.hpp"

using std::vector;
using std::string;
using std::make_pair;

using boost::algorithm::ends_with;

using boost::filesystem::directory_iterator;
using boost::filesystem::is_regular_file;
using boost::filesystem::path;


vector<string>
list_files(const string &dir_path, const vector<string> &extensions)
{
    directory_iterator dir(dir_path), end;
    vector<string> filenames;
    BOOST_FOREACH(const path &p, make_pair(dir, end)) {
        if (!is_regular_file(p))
            continue;

        string filename = p.string();
        for (const string &ext : extensions)
            if (ends_with(filename, ext)) {
                filenames.push_back(filename);
                break;
            }
    }
    return filenames;
}

