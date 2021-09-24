#include <fstream>
#include <iomanip>
#include <string>
#include <iostream>

using namespace std;

void write1(){
    cout << "write1" << endl;
    ofstream file;
    file.open("tests/test1.txt");
    file << 155 << " ";
    file.close();
}

void write2(){
    cout << "write2" << endl;
    ofstream file("tests/test1.txt", ios::app);
    //file.open("tests/test1.txt");
    file << 245;
    file.close();
}

int main(){
    system("mkdir tests");
    write1();
    write2();
    
    
    return 0;
}
