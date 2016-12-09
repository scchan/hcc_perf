//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

/**
 * @file hc_profile.hpp
 * Profiling infrastructure
 */


#pragma once

#include <string>
#include "CXLActivityLogger.h"

#define __HC_XSTR(S)   __HC_STR(S)
#define __HC_STR(S)    #S
#define HC_SCOPE_MARKER amdtScopedMarker(  (std::string(__FUNCTION__) \
                                           + std::string(" ") \
                                           + std::string(__HC_XSTR([__FILE__:__LINE__]))).c_str() \
                                           ,nullptr,nullptr );

