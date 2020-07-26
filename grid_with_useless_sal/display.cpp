#include "Saliency.h"  
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <numeric> 
#include <assert.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
using namespace std;
using namespace cv;

vector<Point> sal_max_n_coords(Mat aMat, int first_n, int grid_row, int grid_col, const int grid_size, const int rows, const int cols);

template <typename T>
vector<int> sort_indexes(const vector<T> &v);

/* sort template, keep indexes, eg. [4,3,5,6]->[1,0,2,3] */
template <typename T>
vector<int> sort_indexes(const vector<T> &v) {
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
  return idx;
}

/* divide the scene according to grid_size, obtain the saliency map, find the max n cells that have the max n mean value */
vector<Point> sal_max_n_coords(Mat aMat, int first_n, int grid_row, int grid_col, const int grid_size, const int rows, const int cols)
{

        cout << "[Info]img size: " << rows << "x" << cols << endl; 
        cout << "[Info]grid size: " << grid_size << endl; 
        cout << "[Info]divide into: " << grid_row << "x" << grid_col << endl;
        vector<float> mean_value;
        // row first
        for(int r = 0; r < grid_row; r++ )
        {
            for(int c = 0; c < grid_col; c++ )
            {
                int lt_x = c*grid_size; // lt pt will never exceed the grid
                int lt_y = r*grid_size;
                int increment_x = lt_x+grid_size>cols?cols-c*grid_size:grid_size; // redundant part processing
                int increment_y = lt_y+grid_size>rows?rows-r*grid_size:grid_size; // redundant part processing               
                // cout << endl << lt_x << "," << lt_y << ","<< lt_x+increment_x << "," << lt_y+increment_y << endl;                                
                mean_value.push_back(mean(aMat(Rect(lt_x, lt_y, increment_x, increment_y)))[0]);
            }
        }     
        // in case u want to check the each value of the grid, row first
        cout << "[Info]mean brightness value of gird(row first): " << endl; 
        int format_counter = 0;
        for (auto iter : mean_value)
        { 
            format_counter += 1;
            cout << setiosflags(ios::fixed) << setprecision(4) << iter << " "; 
            if (format_counter % grid_col == 0)
                cout << endl;
        }
        vector<int> sort_idx = sort_indexes(mean_value); // descend

        vector<Point> coords; // push left top coords here
        for(int i = 0; i < first_n; i++){
            //cout << "value of index_vec [" << i << "] = " << sort_idx[i] << endl;
            int coord_y = (int)(sort_idx[i]/grid_col)*grid_size; // draw some grids to understand the equation 
            int coord_x = (int)(sort_idx[i]%grid_col)*grid_size;
            coords.push_back(Point(coord_x, coord_y)); // push into
            rectangle(aMat, Point(coord_x,coord_y), Point(coord_x+grid_size,coord_y+grid_size), Scalar(255,0,0), 1); // draw the rect
            //cout << "left top coord of index_vec [" << i << "] = " << coord_x << "," << coord_y << endl;
        }      

        cout << "[Info]" << coords.size() << " grids selected" <<endl;  // the size should be equal to the arg first_n        
        //waitKey();
        return coords;

}

int main(int argc, char** argv)
{
    Saliency sal;


    double t = (double)getTickCount();

    Mat src = imread(argv[1]);
    // halfed
    resize(src, src, Size(src.cols/2, src.rows/2), 0, 0, INTER_LINEAR); 
    if (src.empty()) return -1;
 
    vector<unsigned int >imgInput;
    vector<double> imgSal;
 
    //Mat to vector
    int nr = src.rows; // number of rows  
    int nc = src.cols; // total number of elements per line  
    if (src.isContinuous()) {
        // then no padded pixels  
        nc = nc*nr;
        nr = 1;  // it is now a 1D array  
    }
 
    for (int j = 0; j<nr; j++) {
        uchar* data = src.ptr<uchar>(j);
        for (int i = 0; i<nc; i++) {
            unsigned int t = 0;
            t += *data++;
            t <<= 8;
            t += *data++;
            t <<= 8;
            t += *data++;
            imgInput.push_back(t);
 
        }                
    }
 
    sal.GetSaliencyMap(imgInput, src.cols, src.rows, imgSal);
 
    //vector to Mat
    int index0 = 0;
    Mat imgout(src.size(), CV_64FC1);
    for (int h = 0; h < src.rows; h++) {
        double* p = imgout.ptr<double>(h);
        for (int w = 0; w < src.cols; w++) {
            *p++ = imgSal[index0++];
        }
    }
    normalize(imgout, imgout, 0, 1, NORM_MINMAX);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "[Info]saliency cost: " << t << endl; 

    // imshow("raw", src);
    // imshow("sal", imgout);
    double t2 = (double)getTickCount();
    // grid cell size
    const int grid_size = 100;
    // get rows, cols to divide the scene
    const int rows = imgout.rows;
    const int cols = imgout.cols;
    // redundant part will not fill a whole cell, expand another cell
    int grid_row = (int)(rows/grid_size)+1;
    int grid_col = (int)(cols/grid_size)+1;
    assert(grid_row);assert(grid_col); 
    // second arguement represents max N
    vector<Point> coords_sal = sal_max_n_coords(imgout, 10, grid_row, grid_col, grid_size, rows, cols); // sal

    t2 = ((double)getTickCount() - t2)/getTickFrequency();
    cout << "[Info]max " << coords_sal.size() << " selection cost: " << t2 << endl; 
    cout << "[Info]max " << coords_sal.size() <<  " coords: ";     
    for (auto iter_pt : coords_sal)
    {
        cout << "(" << iter_pt.x << "," << iter_pt.y << ")" << " ";
    }
    cout << endl;
    // imshow("sal", imgout);
    // waitKey(0);
 
    return 0;
}
