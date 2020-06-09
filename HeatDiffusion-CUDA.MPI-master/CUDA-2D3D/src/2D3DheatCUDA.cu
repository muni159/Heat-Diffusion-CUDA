#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cuda.h>
using namespace std;


__global__ void Dim2_Calculation(float * __restrict__ d_tem_res, float * __restrict__ d_tem_meo, 
    const float * __restrict__ d_tem_fix, const int width, const int height, const float k)
{
    const int curId = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = curId/height;
    const int h = curId - w*height;
    int top, down, left, right;
    if( w<width && h<height){
        //float tmp = 0;
        if (w>0){
            //printf("(w-1)*height+h = %d\n", (w-1)*height+h);
            //tmp += k * ( - d_tem_meo[curId] + d_tem_meo[(w-1)*height+h]);
            left = w-1;
        }else{
            left=w;
        }
        if (w<width-1){
            //printf("(w+1)*height+h = %d\n", (w+1)*height+h);
            //tmp += k * ( - d_tem_meo[curId] + d_tem_meo[(w+1)*height+h]);
            right = w+1;
        }else{
            right = w;
        }
        if (h>0){
            //printf("(w)*height+h-1 = %d\n", (w)*height+h-1);
            //tmp += k * ( - d_tem_meo[curId] + d_tem_meo[w*height+(h-1)]);
            top = h-1;
        }else{
            top = h;
        }
        if (h<height-1){
            //printf("(w)*height+h+1 = %d\n", (w)*height+h+1);
            //tmp += k * ( - d_tem_meo[curId] + d_tem_meo[w*height+(h+1)]);
            down = h+1;
        }else{
            down = h;
        }
        d_tem_res[curId] = d_tem_meo[curId]+k*(d_tem_meo[left*height+h]+d_tem_meo[right*height+h]
            + d_tem_meo[w*height+top] + d_tem_meo[w*height+down] - 4*d_tem_meo[curId]);
    }
    if ( d_tem_fix[curId] != -1){
        d_tem_res[curId] = d_tem_fix[curId];
    }
    // wait until other thread if finished.
    __syncthreads();
    d_tem_meo[curId] = d_tem_res[curId];

}


__global__ void Dim3_Calculation(float * __restrict__ d_tem_res, float * __restrict__ d_tem_meo, 
    const float * __restrict__ d_tem_fix, const int width, const int height, const int depth, const float k)
{
    const int curId = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = curId/(height*width);
    const int w = (curId-d*(height*width))/height;
    const int h = curId-d*(height*width) - w*height;
    int top, down, left, right, front, back;
    if( w<width && h<height && d<depth){
        if (w>0){
            left = w-1;
        }else{
            left=w;
        }
        if (w<width-1){
            right = w+1;
        }else{
            right = w;
        }
        if (h>0){
            top = h-1;
        }else{
            top = h;
        }
        if (h<height-1){
            down = h+1;
        }else{
            down = h;
        }
        if (d>0){
            front = d-1;
        }else{
            front = d;
        }
        if (d<depth-1){
            back = d+1;
        }else{
            back = d;
        }
        d_tem_res[curId] = d_tem_meo[curId]+k*(d_tem_meo[d*(height*width)+left*height+h]+d_tem_meo[d*(height*width)+right*height+h]
            + d_tem_meo[d*(height*width)+w*height+top] + d_tem_meo[d*(height*width)+w*height+down] 
            + d_tem_meo[front*(height*width)+w*height+h] + d_tem_meo[back*(height*width)+w*height+h] - 6*d_tem_meo[curId]);
        
    }
    if ( d_tem_fix[curId] != -1){
        d_tem_res[curId] = d_tem_fix[curId];
    }
    // wait until other thread if finished.
    __syncthreads();
    d_tem_meo[curId] = d_tem_res[curId];

}





int main(int argc,char**argv)
{
// ------------------------------------ initial parameter --------------------------------------
    string Dimension;
    string path = argv[1];
    ifstream cfile(path.c_str());
    int timestep, width, height, depth, totalLength;
    int location_x, location_y, location_z, fix_width, fix_height, fix_depth;
    float init_temp, ftemp, k;
    float *tem_res, *tem_meo, *tem_fix;
    float *d_tem_res, *d_tem_meo, *d_tem_fix; 

// ------------------------------------Reading the config file ---------------------------------------

    string l;
    vector<string> fileContent;
    while(getline(cfile, l))
    {
    	if(l[0] != '#' && !l.empty())
		{fileContent.push_back(l);}
	}
        
    // -------------------------Dimension, k, timestep, init_temp---------------------------
    Dimension = fileContent[0];
    if(fileContent[1][0]=='.')
    {fileContent[1].insert(fileContent[1].begin(), '0');}
    k = (float)atof(fileContent[1].c_str());
    timestep = atoi(fileContent[2].c_str());
    init_temp = (float)atof(fileContent[4].c_str());
    cout << "k=" << k << " timestep=" << timestep << " BeginTemperture=" << init_temp << endl;

// --------------------begin calculation based on Dimension----------------------
	
	// read width and height, then build the matrix.
    if(Dimension=="2D")
    {
        string::size_type pos = fileContent[3].find(",");
        width = atoi(fileContent[3].substr(0, pos).c_str());
        height = atoi(fileContent[3].substr(pos+1).c_str());
        cout << "Width=" << width << " Height=" << height << endl;
        totalLength = width*height;
    }
    else{
    	string::size_type pos = fileContent[3].find(",");
    	string::size_type pos2 = fileContent[3].find_last_of(",");
        width = atoi(fileContent[3].substr(0, pos).c_str());
        height = atoi(fileContent[3].substr(pos+1,pos2-pos-1).c_str());
        depth = atoi(fileContent[3].substr(pos2+1).c_str());
        cout << "Width=" << width << " Height=" << height << " Depth="<< depth << endl;
        totalLength = width*height*depth;
    }
    tem_res = (float *)malloc(totalLength * sizeof(float));
    tem_meo = (float *)malloc(totalLength * sizeof(float));
    tem_fix = (float *)malloc(totalLength * sizeof(float));
    for (int i = 0; i < totalLength; ++i)
    {
        tem_res[i] = init_temp;
        tem_meo[i] = init_temp;
        tem_fix[i] = -1;
    }

// ------------------------ initialize matrix ----------------------------------
    if(Dimension=="2D")
    {
        for(int i=5; i<fileContent.size(); i++)
        {
            string s = fileContent[i];
            location_x = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            location_y = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            fix_width = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            fix_height = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            ftemp = (float)atof(s.c_str());

            for(int w=location_x; w<location_x+fix_width; w++)
            {
                for(int h=location_y; h<location_y+fix_height; h++)
                {
                    tem_res[w*height+h] = ftemp;
                    tem_meo[w*height+h] = ftemp;
                    tem_fix[w*height+h] = ftemp;
                }
            }
        }
        //for (int i=0; i<totalLength; i++){cout << tem_fix[i] << endl;}
    }
    else{
    	for(int i=5; i<fileContent.size(); i++)
        {
            string s = fileContent[i];
            location_x = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            location_y = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            location_z = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            fix_width = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            fix_height = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            fix_depth = atoi(s.substr(0,s.find(",")).c_str());
            s.erase(0,s.find(",")+1);
            ftemp = (float)atof(s.c_str());

            for(int w=location_x; w<location_x+fix_width; w++)
            {
                for(int h=location_y; h<location_y+fix_height; h++)
                {
                    for(int d=location_z; d<location_z+fix_depth; d++)
                    {
                        tem_res[d*width*height + w*height + h] = ftemp;
                    	tem_meo[d*width*height + w*height + h] = ftemp;
                    	tem_fix[d*width*height + w*height + h] = ftemp;
                    }
                }
            }
        }
        //for (int i=0; i<totalLength; i++){cout << tem_fix[i] << endl;}
    }

	cudaMalloc((void **)&d_tem_res, totalLength * sizeof(float));
	cudaMalloc((void **)&d_tem_meo, totalLength * sizeof(float));
	cudaMalloc((void **)&d_tem_fix, totalLength * sizeof(float));
    cudaMemcpy(d_tem_res, tem_res, totalLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tem_meo, tem_meo, totalLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tem_fix, tem_fix, totalLength * sizeof(float), cudaMemcpyHostToDevice);

// ----------------------- call Cuda Calculation -----------------------------------
    cout << "You begin Cuda Calculation" << endl;
    int bknum = 128;
    if(Dimension=="2D")
    {   
        for (int i = 0; i < timestep; i++) 
        {   
            //cout << "Cuda Calculation " << i <<endl;
        	Dim2_Calculation <<<(totalLength+bknum-1)/bknum, bknum>>>(d_tem_res, d_tem_meo, d_tem_fix, width, height, k);
        }
        cudaMemcpy(tem_res, d_tem_res, totalLength*sizeof(float), cudaMemcpyDeviceToHost);
    }
    else
    {
        for (int i = 0; i < timestep; i++) 
        {
        	Dim3_Calculation <<<(totalLength+bknum-1)/bknum, bknum>>>(d_tem_res, d_tem_meo, d_tem_fix, width, height, depth, k);
        }
        cudaMemcpy(tem_res, d_tem_res, totalLength*sizeof(float), cudaMemcpyDeviceToHost);
    }


// ------------------------- Writing into output.csv -------------------------------
    ofstream result;
    result.open("heatOutput.csv");
    if(Dimension=="2D")
    {
        for (int w = 0; w < width; w++) 
        {
            for (int h = 0; h < height; h++) 
            {	
            	result << tem_res[w*height+h];
                if(h<height-1)
                {result << ",";}
            }
            result << '\n';
        }
    }
    else
    {
        for(int d=0; d<depth; d++)
        {
            for (int w = 0; w < width; w++) 
            {
                for (int h = 0; h < height; h++) 
                {	
                	result << tem_res[d*width*height+w*height+h];
                	if(h<height-1)
                	{result << ",";}
                }
                result << '\n';
            }
            result << '\n';
        }        
    }
    result.close();

// -------------------- free meomery -------------------
    cudaFree(d_tem_res); 
    cudaFree(d_tem_meo); 
    cudaFree(d_tem_fix); 
    free(tem_res); 
    free(tem_meo); 
    free(tem_fix); 

	return 0;
}