// set up the mutation observer
var observer = new MutationObserver(function (mutations, me) {
  // `mutations` is an array of mutations that occurred
  // `me` is the MutationObserver instance
  var rst_current_version = $('.rst-current-version');
  if (rst_current_version) {
    $('.rst-current-version .fa-caret-down').toggleClass(['fa-caret-up', 'fa-caret-down'])

    $('.rst-current-version').click(function() {
      $('.rst-current-version .fa-caret-down, .rst-current-version .fa-caret-up')
        .toggleClass(['fa-caret-up', 'fa-caret-down']);
      $('.rst-other-versions').toggle();
    });

    me.disconnect(); // stop observing
    return;
  }
});

// start observing
observer.observe(document, {
  childList: true,
  subtree: true
});
