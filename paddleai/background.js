var last_word = '';
var count = 0;


var reader = new XMLHttpRequest() || new ActiveXObject('MSXML2.XMLHTTP');

function loadFile() {
    reader.open('get', 'out.txt', false); 
    reader.onreadystatechange = displayContents;
    reader.send(null);
}

function displayContents() {
    if(reader.readyState==4) {
        var total_string = reader.responseText;
		var last_line = total_string.split("\n");
		var last_line = last_line[last_line.length-1];
		if(last_word !== last_line){
			chrome.browserAction.setBadgeText({"text": (++count).toString()});
			new Notification("You have a lot of important messages!", {
		    	icon: '48.png',
				body: last_line
			});
			last_word = last_line;
		}
    }
}

chrome.alarms.onAlarm.addListener(function(alarm){
	loadFile();
});

chrome.alarms.create("alarm", {delayInMinutes: 0.01, periodInMinutes: 0.01});

chrome.tabs.onSelectionChanged.addListener(function(tabId, props) {
  selectedId = tabId;
});

chrome.browserAction.onClicked.addListener(function(tabId, props) {
	count = 0;
	chrome.browserAction.setBadgeText({"text": count.toString(), tabId: selectedId});  

    reader.open('get', 'out.txt', false); 
    reader.onreadystatechange = function(){
		if(reader.readyState==4) {
        	alert(reader.responseText);
		}
	};
    reader.send(null);

});
