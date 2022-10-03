/* 
sphinx-book-theme announcements are pure html, 
so cannot use sphinx :ref: for output destination URIs relative to the current page. 
Instead, find the appropriate link within the current page, to add the anchor to the announcement.
This does not currently support linking outside of the documentation website.
*/
console.log("foo")
window.onload = function () {
    // Only apply the link if the annoucement content matches (incase the announcement is changed but this file is not.)
    expectedAnnouncementContent = "MFA Deployment in Progress";
    // The label for the TOC navigation link which is to be linked against, i.e. the top level title in the page.
    expectedAnchorContent = "Using Bede";
    // If linking to a specific sub-section via sphinx generated ID, manually specify it here.
    optionalTargetID = "#multi-factor-authentication";
        
    // Remove a potential initial # from the above URI, to make latter processing simpler.
    // This could be made more robust, in case of multiple hashes.
    if (optionalTargetID && optionalTargetID.startsWith("#")) {
        optionalTargetID = optionalTargetID.slice(1);
    }
    var elements = document.getElementsByClassName("announcement");
    for (var i = 0; i < elements.length; i++) {
        var element = elements.item(i);
        originalContent = element.innerHTML;
        if(originalContent.toLowerCase().includes(expectedAnnouncementContent.toLowerCase())) {
            // Iterate the TOC until the link with the expected content is found for the requested page.
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
}