<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border:
inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed
'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript"
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

# Machine Learning for Complete Intersection Calabi-Yau 3-folds

We consider a machine learning approach to the issue related to the prediction
of the **Hodge numbers** ($h_{11}$ and $h_{21}$) of **Calabi-Yau 3-folds**.

 We consider several algorithms in order to reproduce them through a
**regression** task. We also implement some **convolutional neural networks** to
help the predictions and ultimately stack them in order to produce the best
result.
