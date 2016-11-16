

#ifndef _OPENCLKERNELHELPER_H
#define _OPENCLKERNELHELPER_H

#define OPENCL_DEFINE(VAR,...)  "\n #""define " #VAR " " #__VA_ARGS__ " \n"
#define OPENCL_ELIF(...)        "\n #""elif " #__VA_ARGS__ " \n"
#define OPENCL_ELSE()           "\n #""else " " \n"
#define OPENCL_ENDIF()          "\n #""endif " " \n"
#define OPENCL_IF(...)          "\n #""if " #__VA_ARGS__ " \n"
#define OPENCL_PRAGMA(VAR,...)  "\n #""pragma " #VAR " " #__VA_ARGS__ " \n"
#define STRINGIFY(...)          #__VA_ARGS__ "\n"
#define STRINGIFYNNL(...)       #__VA_ARGS__

#endif     /* #ifndef _OPENCLKERNELHELPER_H */
