from django import forms
from django.forms.widgets import HiddenInput


class DeepFeatureForm(forms.Form):
    learning_rate = forms.FloatField(label='Learning Rate', initial=0.1)
    layer_index = forms.IntegerField(label='Layer Index', initial=2)
    image_std_clip = forms.FloatField(label='Image Std Clip', initial=3)
    grad_std_clip = forms.FloatField(label='Gradient Std Clip', initial=1.5)
    epoch_count = forms.IntegerField(label='Number of Epochs', initial=250)
    total_variation = forms.FloatField(label='Total Variation Coefficient', initial=0)
    noise_count = forms.IntegerField(label='Noise Count', initial=0)

    def __init__(self, *args, **kwargs):
        super(DeepFeatureForm, self).__init__(*args, **kwargs)
        self.fields['layer_index'].widget = HiddenInput()