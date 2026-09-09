# Basic Usage

You can create a model using the {py:func}`BinaryModel.from_file <lcurve.BinaryModel.from_file>` method, which creates a {py:obj}`BinaryModel <lcurve.BinaryModel>` instance.

```python
import lcurve
binary_model = lcurve.BinaryModel.from_file("WDdM.mod")
```

## Random examples

:::{tip}
Let's give readers a helpful hint!
:::

Sometimes we need to include maths, like $E=mc^2$ or

$$
(a + b)^2  &=  (a + b)(a + b) \\
           &=  a^2 + 2ab + b^2
$$ (mymath2)

The equation {eq}`mymath2` is also a quadratic equation.

Using `sphinx.ext.intersphinx` we can link to other projects' documentation. For example, the Python `random` module: {py:mod}`random` or the Astropy `Time` class: {py:class}`astropy.time.Time`.
