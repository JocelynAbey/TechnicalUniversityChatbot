from django.db import models

class ChatHistory(models.Model):
    question = models.TextField()
    answer = models.TextField()
    confidence = models.FloatField()
    category = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.question[:50]}..."