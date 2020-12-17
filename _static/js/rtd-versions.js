// set up the mutation observer
var observer = new MutationObserver(function (mutations, me) {
  // `mutations` is an array of mutations that occurred
  // `me` is the MutationObserver instance
  var rst_current_version = $('.rst-current-version');
  if (rst_current_version.length > 0) {
    $('.rst-current-version .fa-caret-down').removeClass('fa-caret-down').addClass('fa-caret-up');

    $('.rst-current-version').click(function() {
      if ($('.rst-current-version .fa-caret-down').length > 0) {
        $('.rst-current-version .fa-caret-down').removeClass('fa-caret-down').addClass('fa-caret-up');
      } else {
        $('.rst-current-version .fa-caret-up').removeClass('fa-caret-up').addClass('fa-caret-down');
      }
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
