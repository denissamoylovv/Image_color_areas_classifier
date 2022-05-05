function send_and_receive_image() {
	var formData = $('form[name=input]').serialize();

	var file = $("#image")[0].files[0];
	getBase64(file).then( function (result){
		formData += "&image_str=" + result;
		// console.log(result)

		$.ajax({
			type: "POST",
			url: "/",
			data: formData,
			success: function(data, status) {
				console.log(status);
				var img = $("img#output_image");
				img.attr('src', 'data:image/png;base64,' + data);

				// console.log(data);
			},
			error: function(error) {
				console.log(error);
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
  
  