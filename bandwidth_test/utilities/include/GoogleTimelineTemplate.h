
#include "openclkernelhelper.h"


#ifdef __cplusplus
extern "C" {
#endif

const char* google_html_template = STRINGIFYNNL(
<head>
<script type=\x22text/javascript\x22 src=\x22https:\/\/www.google.com/jsapi?autoload={'modules':[{'name':'visualization',
       'version':'1','packages':['timeline']}]}\x22></script>
<script type=\x22text/javascript\x22>
       google.setOnLoadCallback(drawChart);
       function drawChart() {
         var container = document.getElementById('example1');
         var chart = new google.visualization.Timeline(container);
         var dataTable = new google.visualization.DataTable();
<TIMELINE_CHART_DATA>
           chart.draw(dataTable);
       }
</script>
</head>
<body>
<div id = \x22)STRINGIFYNNL(example1\x22 style = \x22width: 1280px; height: 480px; \x22>< / div>
</body>
</html>
);



const char* timeline_template = STRINGIFYNNL(

<html>
<head>
    <title>Timeline demo</title>

    <style type=\x22text/css\x22>
        body {font: 10pt arial;}
    </style>

    <script type=\x22text/javascript\x22 src=\x22http://www.google.com/jsapi\x22></script>
    <script type=\x22text/javascript\x22 src=\x22../timeline.js\x22></script>
    <link rel=\x22stylesheet\x22 type=\x22text/css\x22 href=\x22../timeline.css\x22>

    <script type=\x22text/javascript\x22>
        var timeline;

        google.load(\x22visualization\x22, \x221\x22);

        // Set callback to run when API is loaded
        google.setOnLoadCallback(drawVisualization);

        // Called when the Visualization API is loaded.
        function drawVisualization() {
            // Create and populate a data table.
            var data = new google.visualization.DataTable();
            data.addColumn('datetime', 'start');
            data.addColumn('datetime', 'end');
            data.addColumn('string', 'content');
            
<TIMELINE_CHART_DATA>

            // specify options
            var options = {
                \x22width\x22:  \x22 100%\x22,
                \x22height\x22: \x22 300px\x22,
                \x22style\x22: \x22box\x22
            };

            // Instantiate our timeline object.
            timeline = new links.Timeline(document.getElementById('mytimeline'));

            // Draw our timeline with the created data and options
            timeline.draw(data, options);
        }
    </script>
</head>

<body>
<div id=\x22mytimeline\x22></div>

<!-- Information about where the used icons come from -->
<p style=\x22color:gray; font-size:10px; font-style:italic;\x22>
    Icons by <a href=\x22http://dryicons.com\x22 target=\x22_blank\x22 title=\x22Aesthetica 2 Icons by DryIcons\x22 style=\x22color:gray;\x22 >DryIcons</a>
    and <a href=\x22http://www.tpdkdesign.net\x22 target=\x22_blank\x22 title=\x22Refresh Cl Icons by TpdkDesign.net\x22 style=\x22color:gray;\x22 >TpdkDesign.net</a>
</p>

</body>
</html>


);


const char* visjs_timeline_template = STRINGIFYNNL(

<!doctype html>
<html>
<head>
  <title>Timeline | Basic demo</title>
  <script src="http://visjs.org/dist/vis.js"></script>
  <link href="http://visjs.org/dist/vis.css" rel="stylesheet" type="text/css" />

  <style type="text/css">
    body, html {
      font-family: sans-serif;
    }
  </style>
</head>
<body>
<div id="mytimeline"></div>
<p></p>
<div id="log"></div>

<script type="text/javascript">
  var container = document.getElementById('mytimeline');
 
<TIMELINE_CHART_DATA>

  var options = {
    stack: false
  };
  var timeline = new vis.Timeline(container, data, options);

  timeline.on('select', function (properties) {
    logEvent('select', properties);
  });

  items.on('*', function (event, properties) {
    logEvent(event, properties);
  });

  function logEvent(event, properties) {
    var log = document.getElementById('log');
    var msg = document.createElement('div');
    msg.innerHTML = 'event=' + JSON.stringify(event) + ', ' +
        'properties=' + JSON.stringify(properties);
    log.firstChild ? log.insertBefore(msg, log.firstChild) : log.appendChild(msg);
  }

</script>
</body>
</html>


);

#ifdef __cplusplus
}
#endif

