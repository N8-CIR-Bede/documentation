// The sphinx-bootstrap-theme handles the {{ toctree() }} content generates different markup for global and local toctree content. This JS applies the `current` css class to list items for internal references on page load and if any internal links are clicked.
$(document).ready(function() {
    // On page load, mark localtoc elements as current if appropriate
    $('.bs-sidenav .nav li > a.reference.internal').each(function() {
        if (this.href === window.location.href) {
            $(this).parent().addClass('current');
            $(this).parents('.bs-sidenav li').addClass('current')
        }
    });
    // on click of an internal reference in the toctree, adjust use of the current css class in the sidebar as appropriate
    $('.bs-sidenav .nav li > a.reference.internal').click(function() {
        // Remove the current class from others
        $('.nav li').has("a.reference.internal").removeClass('current');
        // Mark the new selected link as current.
        $(this).parent().addClass("current");
        // Mark parents 
        $(this).parents('.bs-sidenav li').addClass('current')

    });
});