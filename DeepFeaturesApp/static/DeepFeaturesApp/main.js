var layers = [
    ['block1_conv1', '(224, 224, 64)', '1792'],
    ['block1_conv2', '(224, 224, 64)', '36928'],
    ['block1_pool', '(112, 112, 64)', '0'],
    ['block2_conv1', '(112, 112, 128)', '73856'],
    ['block2_conv2', '(112, 112, 128)', '147584'],
    ['block2_pool', '(56, 56, 128)', '0'],
    ['block3_conv1', '(56, 56, 256)', '295168'],
    ['block3_conv2', '(56, 56, 256)', '590080'],
    ['block3_conv3', '(56, 56, 256)', '590080'],
    ['block3_pool', '(28, 28, 256)', '0'],
    ['block4_conv1', '(28, 28, 512)', '1180160'],
    ['block4_conv2', '(28, 28, 512)', '2359808'],
    ['block4_conv3', '(28, 28, 512)', '2359808'],
    ['block4_pool', '(28, 28, 256)', '0'],
    ['block5_conv1', '(14, 14, 512)', '2359808'],
    ['block5_conv2', '(14, 14, 512)', '2359808'],
    ['block5_conv3', '(14, 14, 512)', '2359808'],
    ['block5_pool', '(7, 7, 512)', '0'],
    ['flatten', '(25088)', '0'],
    ['fc_1', '(4096)', '0'],
    ['fc_2', '(4096)', '0'],
    ['predictions', '(1000)', '0']
];

$(document).ready(function() {
    // load a preview of the custom image selected
    document.getElementById('custom-image-input').onchange = function (evt) {
        var tgt = evt.target || window.event.srcElement,
            files = tgt.files;
        // FileReader support
        if (FileReader && files && files.length) {
            var fr = new FileReader();
            fr.onload = function () {
                document.getElementById('custom-image-preview').src = fr.result;
                $("#save-custom-image").prop("disabled", false);
            }
            fr.readAsDataURL(files[0]);
        }
        // Not supported
        else {
            $("#save-custom-image").prop("disabled", true);
        }
    }

    // make the neural network layers selectable
    $(".selectable").click(function(e) {
        $(".selectable").removeClass("selected");
        $(e.target).addClass("selected");

        var layerId = $(e.target).attr('id');
        console.log(layerId);
        var layerIndex = parseInt(layerId.replace(/[^\d]/g, ''));
        console.log(layerIndex);

        $("#id_layer_index").val(layerIndex.toString());
        console.log($("#id_layer_index").val());

        var title = layers[layerIndex - 1][0];
        $("#layer-title").text(title);

        var params = layers[layerIndex - 1][1] + " " + layers[layerIndex - 1][2] + " params";
        $("#layer-params").text(params);
    });

    $(".use-params").click(function (e) {
        var cell = e.target.parentElement;

        var historyIndex = parseInt($(cell).find("#index").get(0).value);
        var historyIndexReverse = -historyIndex - 1;
        $.ajax({
            type: "GET",
            url: "historyParams?index=" + historyIndexReverse.toString(),
            success: function (data) {
                console.log(data);
                var history = JSON.parse(data);

                $("#id_learning_rate").val(history.learning_rate);
                $("#id_layer_index").val(history.layer_index);
                $("#id_image_std_clip").val(history.image_std_clip);
                $("#id_grad_std_clip").val(history.grad_std_clip);
                $("#id_epoch_count").val(history.epoch_count);
                $("#id_total_variation").val(history.total_variation);

                // select the correct layer on the diagram
                var layer_selector = "#layer" + history.layer_index;
                $(".selectable").removeClass("selected");
                $(layer_selector).addClass("selected");

                $("#image-gen-form").find(".form-control").addClass("shadow-primary");
                setTimeout(function() {
                    $("#image-gen-form").find(".form-control").removeClass("shadow-primary");
                }, 500);
            }
        });
    });

    if ($("#image-art").length == 0)
        return;

    var imagePath = $("#image-art").attr("rawsrc");
    var refreshIntervalId = setInterval(function() {
        var imagePathFresh = imagePath + "?" + new Date().getTime();

        $.ajax({
            type: "GET",
            url: imagePathFresh,
            success: function (data) {

                $("#image-art").attr("src", imagePathFresh);
                $("#image-art").removeAttr("hidden");
                $("#loading-container").attr("hidden", "hidden");
                clearInterval(refreshIntervalId);
            }
        });
    }, 2000);
});