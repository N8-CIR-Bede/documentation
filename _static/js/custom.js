/* 
sphinx-book-theme announcements are pure html, 
so cannot use sphinx :ref: for output destination URIs relative to the current page. 
Instead, find the appropriate link within the current page, to add the anchor to the announcement.
*/
window.onload = function () {
    expectedAnnouncementContent = "RHEL 8 Migration in Progress";
    expectedAnchorContent = "RHEL 8 Migration";
    var elements = document.getElementsByClassName("announcement");
    for (var i = 0; i < elements.length; i++) {
        var element = elements.item(i);
        originalContent = element.innerHTML;
        if(originalContent.toLowerCase().includes(expectedAnnouncementContent.toLowerCase())) {
            // Grab the link to the RHEL 8 migration page from the toc.
            anchorList = document.querySelectorAll(".toctree-l1>a.reference.internal");
            for (var j = 0; j < anchorList.length; j++) {
                anchor = anchorList[j];
                if(anchor.innerHTML.includes(expectedAnchorContent)) {
                    newAnnouncementContent = '<a href="' + anchor.href + '">' + originalContent + '</a>'
                    element.innerHTML = newAnnouncementContent;
                }
            }
        }
    }
}