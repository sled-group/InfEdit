# InfEdit Handbook (Draft)

# Edit Config

| Parameter | Description |
| --- | --- |
| Source Prompt | prompt used to describe the input image (should include what you want to edit) |
| Positive Prompt & Guidance scale | Classfier-free Guidance setting.  And Positive Prompt can be highlevelly seen as the features in the input image that you want to preserve. |
| Target Prompt | prompt used to describe the output image (shoude be similar with the source prompt,  including what you want to edit to, see examples) |
| Negative Prompt & Guidance scale | Classfier-free Guidance setting. Negative prompt is used to exclude the features that you dont want to generate. |
| Local blend & Thresh | When you only want to edit some part of the whole image, you should use local blend. Local blend should come from a part of the Target prompt, used to describe the area of the image that you want to edit(see example). A higher threshold means a smaller editing area, while a lower threshold means a larger area. |
| Mutual blend & Thresh | When you find the model edit something you want to keep, you should use mutual blend. Mutual blend should come from a common part of the Source prompt and Target prompt, used to describe the area of the image that you want preserve (exactly same, see example). A higher threshold means a smaller area that preserved, while a lower threshold means a larger area. |
| Same part of Local blend & Mutual blend | the same part of local blend and mutual blend represents the thing that you want to partially edit(e.g. postures, views, actions). (See example) |
| Cross replace steps | Cross-attention control steps. A smaller value means weaker control, which means less consistency. |
| Self replace steps | Self-attention control steps. A smaller value means weaker control, which means less consistency. |
| Denoising Model & Strength | The denoising intensity, the smaller the value, the less modifications and higher consistency. Too low intensity may lead to modification failure. |

# Get start

Step 1, First, we can use this window upload an image that we want to edit easily.

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled.png)

## Source prompt & Target Prompt & Guidance Scale

This is the most basic function of our application.

Step 2, We need to use a prompt to describe the input image, as a base for editing like:

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%201.png)

And then, input the target prompt as you like based on your source prompt. You can **change some words**, or **add some words**, or **delete some words** if you like. Here are some samples to illustrate the ability of our model, and we will further explain how the papremeter choices affect the image editing quality in the later parts.

### Change words

Step 3, **Change word “cat” to “tiger”**, (Step 4) and set the local blend as “tiger”, self replace steps as 0.7. (Step 5) After hitting the run button, you can get the result as below.  **We will talk about those settings in the following parts.**

When we set the Guidance scale = 1 as below, we may find that the output image is not “tiger enough”

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%202.png)

In this case, (Step 6 and 7) we can use a larger guidance scale, we say 2, and click the run button. We will find that the effect of edit will become better.

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%203.png)

### Add words

Back to the second step, in this time we **add “red” as the prefix of “cat”**, (as step 3), and (Step 4) set the guidance scale of Target prompt to 2, local blend as “cat”, both of cross and self replace steps as 0.9. (Step 5) After hitting the run button, you can get the result as below. **We will talk about those settings in the following parts.**

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%204.png)

### Delete words

This time we reuse the previous output as our input, and **remove the word “red”** from the previous prompt. In this time, we set the guidance scale of Target prompt to 2, local blend as “cat”, both of cross and self replace steps as 0.9.  After hitting the run button, you can get the result as below. **We will talk about those settings in the following parts.**

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%205.png)

However, we may find the cat is still a little bit red. We can use a larger guidance scale in source prompt, which means edit the source image more(Without negative prompt).

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%206.png)

If we want to remove more red from the source image, in addition to increasing the scale, we can also use nagetive prompts, which represent the thing we don’t want to generate as other stable diffusions models.

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%207.png)

And positive prompt plays the opposite effect. If we input ”red” as our positive prompt, more red will be kept in the generated image.

### Now let's take a closer look at how each parameter works.

## Local & Mutual Blend

### Local blend: Like Prompt2Prompt, we support local blend in our editing, and here is an example.

We reuse the previous input image for test, and we want to edit the cat to a tiger. If we only use source prompt and target prompt, we may find that the whole image will be edited as below.

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%208.png)

And in most of time, we only want to edit some parts of the original image. In this case, we can use “tiger” as our local blend, which means only the part of tiger will be edited.

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%209.png)

If you want to expand the scope of editing, you can reduce the threshold of local blend. Generally speaking, a higher threshold means a smaller area, while a lower threshold means a larger editing area.

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2010.png)

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2011.png)

### Mutual blend: Sometimes you may find that the model edits something you don’t want to edit, and you really want to keep those stuffs, you can use mutual blend.

For example, if we want to change the background of the following example from the grass to a forest, we may use prompts like below:

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2012.png)

However, this time we find that the shepherd dog, which we want to keep, has also been edited. In this case, we may use “shepherd dog” as our mutual blend, and then you will get the following result:

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2013.png)

Similarity, a higher threshold means a smaller area that preserved, while a lower threshold means a larger area.

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2014.png)

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2015.png)

### Cross & Self Replace Steps

The cross replace step number controls the proportion of the cross-attention control steps in the whole diffusion process. Similarly, the self replace step number sontrols the proportion of the self-attention control steps in the whole diffusion process. Increasing either of the two replace step numbers will increase the consistency between the generated image and the source image.

In the following sample, we will change the cat in the picture into a tiger. While setting all the other parameters as defulat, we fist set both the self replace step and the cross replace step  to 0.5 and can get the following output:

![eacdce970fc0946d96ebe6b852ca20f.png](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/eacdce970fc0946d96ebe6b852ca20f.png)

Then we first change the cross replace step by decreasing it to 0.2:

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2016.png)

Then we change the cross replace step to 0.8:

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2017.png)

We can also modify the self replace step to decrease consistency:

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2018.png)

## Advanced use of Blending (TODO)

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2019.png)

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2020.png)

![Untitled](InfEdit%20Handbook%20(Draft)%20ff6b9e111f9a475999d6896a918b2194/Untitled%2021.png)