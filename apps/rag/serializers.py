from rest_framework import serializers


class QuerySerializer(serializers.Serializer):
    query = serializers.CharField(max_length=4096)
    tags = serializers.ListField(
        child=serializers.CharField(max_length=64),
        required=False,
        allow_empty=True,
    )
    language = serializers.CharField(max_length=8, required=False, allow_blank=True)
    document_type = serializers.CharField(max_length=32, required=False, allow_blank=True)
    top_k = serializers.IntegerField(required=False, min_value=1, max_value=20)
