/**
 * @file    Function Load After Plugins Files Loaded
 * @name    Custom
 * @author  Noman
 */

//One Signal App segmentaion
OneSignal.push(function() {
    OneSignal.on('subscriptionChange', function(isSubscribed) {
        console.log('subscribe click');
        if ($('#sidebar_nl_type').val()){
            var tag_type = $('#sidebar_nl_type').val();
            console.log(tag_type + 'post Page');
        }else{
            var tag_type = $('#oneSignal_cat_nl_type').val();
            console.log(tag_type + 'cat Page');
        }

        if(isSubscribed === true){
            console.log('subcribe true');

            OneSignal.sendTags({
                user_type: tag_type,
            }).then(function(tagsSent) {
                // Callback called when tags have finished sending
                console.log('subcribe true');
                console.log("tagsSent" + JSON.stringify(tagsSent));
                //alert('ho gaya');
            });
        }
    })
});
