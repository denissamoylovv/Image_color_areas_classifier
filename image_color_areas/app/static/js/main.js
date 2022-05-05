function send_and_receive_image() {
	var formData = $('form[name=input]').serialize();

	var file = $("#image")[0].files[0];
	getBase64(file).then( function (result){
		formData += "&image_str=" + result;
		// console.log(result)

		// $("#images").attr('src', result);
		$("#input_image").css('background-image', 'url(' + result + ')');
		image=result;



		$.ajax({
			type: "POST",
			url: "/",
			data: formData,
			success: function(data, status) {
				console.log(status);
				var img = $("#output_image");

				img.attr('src', 'data:image/png;base64,' + data);
				// img.css('background-image', 'url(data:image/png;base64,' + data + ')');

				// $("#button")[0].classList.remove("btn-primary");
				// $("#button")[0].classList.add("btn-success");
				// $("#button")[0].classList.remove("btn-danger");

				// console.log(data);
			},
			error: function(error) {
				console.log(error);
				// $("#button")[0].classList.remove("btn-primary");
				// $("#button")[0].classList.add("btn-danger");
				// $("#button")[0].classList.remove("btn-success");
			}
		});

	});
}

function getBase64(file) {
	return new Promise((resolve, reject) => {
	  const reader = new FileReader();
	  reader.readAsDataURL(file);
	  reader.onload = () => resolve(reader.result);
	  reader.onerror = error => reject(error);
	});
}

no_image=false;
image="";

function update_opacity(value) {
	
	if (value > "1") {
		value=-value+2
		if(no_image==false){
			$("#input_image").css('background-image', 'url(img/white.png)');
			no_image=true;
		}
	}else{
		if(no_image==true){
			$("#input_image").css('background-image', 'url(' + image + ')');
			no_image=false;
		}
	}
	$("#output_image").css('opacity', value);
}

