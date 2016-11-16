#ifdef _WIN32
#include <Windows.h>
#include <process.h>

#define GETPID()  _getpid()

#else
#include <sys/time.h>
#include <sys/types.h>
#include <linux/limits.h>
#include <unistd.h>

#define GETPID()  getpid()

#endif

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>
#include <queue>


#include "StackTimer.hpp"
#include "StackTimer.h"

#include "GoogleTimelineTemplate.h"

#define PRINT_POINTER(p) printf("%s:%d %s=%p\n",__PRETTY_FUNCTION__,__LINE__,#p,p);


// Have a global TimerStack object to bootstrap itself
TimerStack defaultTimerStack;
TimerStack* TimerStack::getDefaultTimerStack() { return &defaultTimerStack; }

class TimerEvent {
public:
  TimerEvent(const std::string& name) {
    this->name = name;
    startTime = 0;
    endTime = 0;
    nestedLevel = 0;
  }

  TimerEvent(const TimerEvent& e):name(e.name),startTime(e.startTime)
                                  ,endTime(e.endTime),nestedLevel(e.nestedLevel){}

  void recordStartTime() {
    startTime = getCurrentTime();
  }

  void recordEndTime() {
    endTime = getCurrentTime();
  }

  long long getElapsedTime() {
    return (endTime - startTime);
  }

  ~TimerEvent() {
  }
  std::string name;
  long long startTime;
  long long endTime;
  unsigned int nestedLevel;
private:

  long long getCurrentTime() {
    long long t;
#ifdef _WIN32
    t = GetTickCount64();
#else
    struct timeval s;
    gettimeofday(&s, 0);
    t = (long long)s.tv_sec * (long long)1.0E3 +
      (long long)s.tv_usec / (long long)1.0E3;
#endif
    return t;
  }
};

class TimerStackImpl {
public:

  TimerStackImpl() {
    maxNestedLevel = 0;
    std::stringstream ss;
    ss << (int)GETPID();
    prefix = ss.str();

    enable = 0;
    const char* timer_enable_str = getenv("TIMER_ENABLE");
    if (timer_enable_str!=NULL) {
      if (std::string(timer_enable_str)=="1") {
        enable = 1;
      }
    }
  }

  void StartTimer(const char* name);
  void StopTimer(const char* name);
  void dumpTimerStack();
  void setLogPrefix(const std::string& prefix);
  void setLogPrefix(const char* prefix);
  ~TimerStackImpl();

protected:
  std::stack<TimerEvent*> timerStack;
  unsigned int maxNestedLevel;

  std::queue<TimerEvent*> startQueue;
  std::queue<TimerEvent*> endQueue;
  std::string prefix;
  char enable;

  void dumpTimerStackGoogleTimeline();
};



void TimerStackImpl::dumpTimerStackGoogleTimeline() {

  std::stringstream filename;
  filename << prefix << ".html";
  std::ofstream file;
  file.open(filename.str().c_str(), std::ios::trunc);

  std::stringstream table;

#define ADD_COLUMN(stream,type,id) (stream) << "dataTable.addColumn({ type: '" << (type) << "', id: '" << (id) << "'});" << std::endl;

  ADD_COLUMN(table, std::string("string"), std::string("Nested Level"));
  ADD_COLUMN(table, std::string("string"), std::string("Name"));
  ADD_COLUMN(table, std::string("date"), std::string("Start"));
  ADD_COLUMN(table, std::string("date"), std::string("End"));

  table << "dataTable.addRows([" << std::endl;
#define ADD_ROW(stream,nestedLevel,name,start,end) (stream) << "[ '" << (nestedLevel) << "', '" << (name) \
  << "', new Date(" << (start) << "), new Date(" << (end) << ")]" << "," << std::endl;

  for (unsigned int i = 0; i < startQueue.size(); i++) {
    TimerEvent* e = startQueue.front();
    ADD_ROW(table, e->nestedLevel, e->name, e->startTime, e->endTime);
    startQueue.pop();
    startQueue.push(e);
  }

  table << "]);" << std::endl;

  std::string htmlString = std::string(google_html_template);
  size_t location = htmlString.find("<TIMELINE_CHART_DATA>");
  htmlString.replace(location, std::string("<TIMELINE_CHART_DATA>").size(), table.str());
  file << htmlString;

  file.close();
}

void TimerStackImpl::StartTimer(const char* name) {
  if (!enable)
    return;

  TimerEvent* e = new TimerEvent(std::string(name));
  e->nestedLevel = (unsigned int)timerStack.size();
  maxNestedLevel = (e->nestedLevel > maxNestedLevel) ? e->nestedLevel : maxNestedLevel;
  timerStack.push(e);
  startQueue.push(e);
  e->recordStartTime();
}
void TimerStackImpl::StopTimer(const char* name) {
  if (!enable)
    return;

  TimerEvent* e = timerStack.top();
  e->recordEndTime();
  timerStack.pop();
  endQueue.push(e);
}



class EventAverage {
public:
  EventAverage(std::string name):name(name),num(0),total(0){}
  EventAverage(std::string name, long long value):name(name),num(1),total(value){}
  void add(long long v) {
    total+=v;
    num++;
  }
  double getAverage() {
    if (num==0) 
      return 0.0;
    else
      return (double)total/(double)num;
  }
  std::string name;
  unsigned int num;
  long long total;
};

class EventAverageManager {
public:
  void addEvent(std::string name, long long value) {
    for (std::vector<EventAverage>::iterator iter = data.begin();
        iter != data.end(); iter++) {
      if (iter->name==name) {
        iter->add(value);
        return;
      }
    }
    data.push_back(EventAverage(name,value));
  }
  std::vector<EventAverage> data;
};

void TimerStackImpl::dumpTimerStack() {
  std::stringstream filename;
  filename << prefix << ".log";
  std::ofstream file;
  file.open(filename.str().c_str(), std::ios::trunc);

  EventAverageManager avg;

  while (!startQueue.empty()) {
    TimerEvent* e = startQueue.front();
    for (unsigned int i = 1; i < e->nestedLevel; i++) {
      file << "  ";
    }
    if (e->nestedLevel != 0)
      file << "|_";
    file << e->name << ": " << e->getElapsedTime() << "ms" << std::endl;

    avg.addEvent(e->name, e->getElapsedTime());
    startQueue.pop();
  }

  // print the averages
  file << std::endl;
  file << "Event averages:" << std::endl;
  for (std::vector<EventAverage>::iterator iter = avg.data.begin();
    iter != avg.data.end(); iter++) {
    file << "\t" << iter->name << ": " << iter->getAverage() << "ms" << std::endl;
  }
  file.close();
}

void TimerStackImpl::setLogPrefix(const char* prefix) {
  this->prefix = std::string(prefix);
}


TimerStackImpl::~TimerStackImpl() {
  
  if (enable) {
    dumpTimerStackGoogleTimeline();
    dumpTimerStack();
  }

  // delete all the timer events
  while (!endQueue.empty()) {
    delete endQueue.front();
    endQueue.pop();
  }
}



TimerStack::TimerStack() {
  impl = new TimerStackImpl();
  timer = new Timer("TimerStack",this);
}
TimerStack::~TimerStack() {
  delete timer;
  delete impl;
}
void TimerStack::StartTimer(const char* name) {
  impl->StartTimer(name);
}
void TimerStack::StopTimer(const char* name) {
  impl->StopTimer(name);
}
void TimerStack::dumpTimerStack() {
  impl->dumpTimerStack();
}
void TimerStack::setLogPrefix(const char* prefix) {
  impl->setLogPrefix(prefix);
}


Timer::Timer(const char* name, TimerStack* ts):tsp(ts) {
  this->name = strdup(name);
  tsp->StartTimer(this->name);
}

Timer::~Timer() {
  tsp->StopTimer(name);
  free(this->name);
}


class TimerEventQueueData{
public:
  std::string prefix;
  std::vector<TimerEvent> timers;
};

TimerEventQueue::TimerEventQueue() {  
  data = new TimerEventQueueData();

  std::stringstream ss;
  ss << (int)GETPID();
  data->prefix = ss.str();
}

TimerEventQueue::~TimerEventQueue() {
  //dumpTimeline();
  dump_visjs_Timeline();
  delete data; 
}

unsigned int TimerEventQueue::getNewTimer(const char* name) {
    data->timers.push_back(TimerEvent(std::string(name)));
    return data->timers.size()-1;
}

TimerEvent* TimerEventQueue::getTimerEvent(const unsigned int index) { 
  if (index < data->timers.size())
    return &(data->timers[index]);
  else
    return NULL;
}

void TimerEventQueue::clear()                 { data->timers.clear(); }

unsigned int TimerEventQueue::getNumEvents()  {  return (unsigned int)data->timers.size();  }

double TimerEventQueue::getNumEvents(const char* name) {
  unsigned int numEvents = 0;
  std::string n = std::string(name);
  for (std::vector<TimerEvent>::iterator it = data->timers.begin();
    it != data->timers.end(); it++) {
    if (n == it->name) {
      numEvents++;
    }
  }
  return numEvents;
}


long long  TimerEventQueue::getTotalTime() {
  long long total = 0;
  for (std::vector<TimerEvent>::iterator it = data->timers.begin();
    it != data->timers.end(); it++) {
    total+=it->getElapsedTime();
  }
  return total;
}

long long TimerEventQueue::getTotalTime(const char* name) {
  long long total = 0;
  std::string n = std::string(name);
  for (std::vector<TimerEvent>::iterator it = data->timers.begin();
    it != data->timers.end(); it++) {
    if (n == it->name) {
      total+=it->getElapsedTime();
    }
  }
  return total;
}


double TimerEventQueue::getAverageTime() {
  if (data->timers.size() == 0) 
    return 0.0;
  else
    return (getTotalTime()/(double)data->timers.size());
}

double TimerEventQueue::getAverageTime(const char* name) {
  long long total = 0;
  unsigned int numEvents = 0;
  std::string n = std::string(name);
  for (std::vector<TimerEvent>::iterator it = data->timers.begin();
    it != data->timers.end(); it++) {
    if (n == it->name) {
      numEvents++;
      total+=it->getElapsedTime();
    }
  }
  if (numEvents==0)
    return 0;
  else
    return (double)total/(double)numEvents;
}

void TimerEventQueue::setLogPrefix(const char* prefix) {
  data->prefix = std::string(prefix);
}

void TimerEventQueue::dumpTimeline() {

  std::stringstream ss;

  ss << std::endl << "data.addRows([" << std::endl;

  for (std::vector<TimerEvent>::iterator iter = data->timers.begin();
      iter != data->timers.end(); iter++) {

#define ADD_TIMELINE_ROW(stream,start,end,name) (stream) << "\t [new Date(" << (start) << "),new Date(" << (end) << "), '" << (name) << "']," << std::endl;

    ADD_TIMELINE_ROW(ss,iter->startTime,iter->endTime,iter->name);

  }

  ss << "]);" << std::endl;

  
  std::string htmlString = std::string(timeline_template);
  size_t location = htmlString.find("<TIMELINE_CHART_DATA>");
  htmlString.replace(location, std::string("<TIMELINE_CHART_DATA>").size(), ss.str());

  std::stringstream filename;
  filename << data->prefix << ".html";
  std::ofstream file;
  file.open(filename.str().c_str(), std::ios::trunc);
  file << htmlString;
  file.close();
}


void TimerEventQueue::dump_visjs_Timeline() {

  std::stringstream ss;
  std::stringstream var;

  ss << std::endl << "var data = [" << std::endl;

  unsigned int i = 0;
  for (std::vector<TimerEvent>::iterator iter = data->timers.begin();
      iter != data->timers.end(); iter++,i++) {

#define DEFINE_ITEM(stream,id,elapsed,name) (stream) << std::endl << "var item" << (id) << " = document.createElement('div');" << std::endl\
                                      << "item" << (id) << ".setAttribute('title', '" << (elapsed) << " ms');" << std::endl\
                                      << "item" << (id) << ".setAttribute('style', 'font-size:10pt;');" << std::endl\
                                      << "item" << (id) << ".appendChild(document.createTextNode('" << name << "'));" << std::endl;

#define ADD_VISJS_TIMELINE_ROW(stream,id,start,end) (stream) << "\t {id: " << (id) \
                                      << ", content: item" << (id) \
                                      << ", start: " << (start) \
                                      << ", end: " << (end) \
                                      << "}," << std::endl;

    DEFINE_ITEM(var,i,iter->getElapsedTime(),iter->name);
    ADD_VISJS_TIMELINE_ROW(ss,i,iter->startTime,iter->endTime);
  }

  ss << "];" << std::endl;
  
  std::string htmlString = std::string(visjs_timeline_template);
  size_t location = htmlString.find("<TIMELINE_CHART_DATA>");
  htmlString.replace(location, std::string("<TIMELINE_CHART_DATA>").size(), var.str()+ss.str());

  std::stringstream filename;
  filename << data->prefix << ".html";
  std::ofstream file;
  file.open(filename.str().c_str(), std::ios::trunc);
  file << htmlString;
  file.close();
}


SimpleTimer::SimpleTimer(TimerEventQueue& q, const char* name)  {
  this->q = &q;
  index = q.getNewTimer(name);
  q.getTimerEvent(index)->recordStartTime();
}
SimpleTimer::~SimpleTimer() {
  q->getTimerEvent(index)->recordEndTime();
}

struct stimer_struct {
  Timer* timer;
};

extern "C" {

  STimer timer_start(const char* name) {
    STimer t = (STimer)malloc(sizeof(stimer_struct));
    t->timer = new Timer(name);
    return t;
  }

  void timer_stop(STimer t) {
    delete t->timer;
    free(t);
  }

}
