/* 
sphinx-book-theme announcements are pure html, 
so cannot use sphinx :ref: for output destination URIs relative to the current page. 
Instead, find the appropriate link within the current page, to add the anchor to the announcement.
This does not currently support linking outside of the documentation website.
*/
/* window.onload = function () {
    expectedAnnouncementContent = "";
    expectedAnchorContent = "";
    var elements = document.getElementsByClassName("announcement");
    for (var i = 0; i < elements.length; i++) {
        var element = elements.item(i);
        originalContent = element.innerHTML;
        if(originalContent.toLowerCase().includes(expectedAnnouncementContent.toLowerCase())) {
            // Grab the link to the destination page from the toc.
            anchorList = document.querySelectorAll(".toctree-l1>a.reference.internal");
            for (var j = 0; j < anchorList.length; j++) {
                anchor = anchorList[j];
                if(anchor.innerHTML.includes(expectedAnchorContent)) {
                    // Build the target URI, if the optional target id is truth (not empty)
                    targetDestination = anchor.href;
                    if (optionalTargetID) {
                        // If we are already on the page, sphinx may have added a # to the TOC URI already
                        // If not, we need to add one.
                        if (!targetDestination.endsWith("#")) {
                            targetDestination += "#";
                        }
                        // then append the optionalTargetID, which should no longer begin with a hash.
                        targetDestination += optionalTargetID;
                    }
                    newAnnouncementContent = '<a href="' + targetDestination + '">' + originalContent + '</a>'
                    element.innerHTML = newAnnouncementContent;
                }
            }
        }
    }
} */
