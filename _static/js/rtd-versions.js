$( document ).ready(function() {
  $('.rst-current-version .fa-caret-down').toggleClass(['fa-caret-up', 'fa-caret-down'])

  $('.rst-current-version').click(function() {
    $('.rst-current-version .fa-caret-down, .rst-current-version .fa-caret-up')
      .toggleClass(['fa-caret-up', 'fa-caret-down']);
    $('.rst-other-versions').toggle();
  });
});
